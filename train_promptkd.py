import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    mpu,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset

from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model, parallel_model_map, get_teacher_model

from accelerate import init_empty_weights

from rouge_metric import compute_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    if args.model_type=="qwen" and ds_config['fp16']['enabled']==True:
        import copy
        ds_config['bf16']=copy.deepcopy(ds_config['fp16'])
        ds_config['fp16']['enabled']=False
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=mpu if args.model_parallel else None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler

def setup_teacher_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the teacher model
    model = get_teacher_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        while isinstance(model, DDP):
            model = model.module

        if args.teacher_peft is not None:
            param_groups = get_optimizer_params_peft(args, model)
        else:
            param_groups = get_optimizer_params(args, model)

        # Use AdamW.
        optimizer = AdamW(param_groups, lr=args.teacher_lr, weight_decay=args.weight_decay)
        print_rank(f'Teacher Optimizer = {optimizer.__class__.__name__}')
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None

    if args.model_type=="qwen" and ds_config['fp16']['enabled']==True:
        import copy
        ds_config['bf16']=copy.deepcopy(ds_config['fp16'])
        ds_config['fp16']['enabled']=False
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=mpu if args.model_parallel else None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler

def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")
    return data


def get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits, is_teacher=False, is_base=False):
    with torch.no_grad():
        teacher_model.eval()
        if is_base:
            teacher_outputs = teacher_model.base_model(**model_batch, use_cache=False)
        elif is_teacher:
            teacher_outputs = teacher_model(**model_batch, use_cache=False)
        else:
            teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
        if is_teacher or is_base:
            teacher_logits = teacher_logits
        else:
            teacher_logits = teacher_logits[:, args.prompt_len:, :]
    if (is_teacher and args.teacher_kld_type == "forward") or (is_base and args.base_kld_type == "forward"):
        if args.model_parallel:
            distil_losses = mpu.parallel_soft_cross_entropy_loss(logits.float(), teacher_logits.float()) # Forward KL
            distil_losses = distil_losses.view(-1)
            loss_mask = no_model_batch["loss_mask"].view(-1)
            distil_loss = (distil_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32) #[B, 512, 50257]
            inf_mask = torch.isinf(logits)
            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0) #[B, 512, 50257]
            x = torch.sum(prod_probs, dim=-1).view(-1) #[B * 512]
            mask = (no_model_batch["label"] != -100).int() # [B, 512]
            distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    else:
        if args.model_parallel:
            distil_losses = mpu.parallel_soft_cross_entropy_loss(teacher_logits.float(), logits.float()) \
                            - mpu.parallel_soft_cross_entropy_loss(logits.float(), logits.float()) # Reverse KL
            distil_losses = distil_losses.view(-1)
            loss_mask = no_model_batch["loss_mask"].view(-1)
            distil_loss = (distil_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            probs = F.softmax(logits, dim=-1, dtype=torch.float32) #[B, 512, 50257]
            teacher_inf_mask = torch.isinf(teacher_logits)
            teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32) #[B, 512, 50257]
            teacher_prod_probs = torch.masked_fill(probs * teacher_logprobs, teacher_inf_mask, 0) #[B, 512, 50257]

            inf_mask = torch.isinf(logits)
            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            prod_probs = torch.masked_fill(probs * logprobs, inf_mask, 0) #[B, 512, 50257]

            x = torch.sum(prod_probs - teacher_prod_probs, dim=-1).view(-1) #[B * 512]
            mask = (no_model_batch["label"] != -100).int() # [B, 512]
            distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss

def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences
    
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")

    # print_inspect(model, '*')
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
        loss_func = mpu.parallel_cross_entropy
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
    best_eval = 0.0
    
    #evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device)
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        #model.train()
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            # torch.save((model_batch, no_model_batch), "mb_few.pt")
            # exit(0)
            torch.cuda.synchronize()
            st_time = time.time()

            # if it == 0 and dist.get_rank() == 0:
            #     torch.save((model_batch, no_model_batch), os.path.join(args.save, "examples.pt"))
            
            ### 1. Sampling from Student ###
            model.eval()
            with torch.no_grad():
                max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens) # max_length로 설정 안하고 그냥 max_new_tokens 직접 넣어서 계산함.
                full_ids = gen_out.sequences            
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]), # 오른쪽에 다시 max_length 만큼 padding. (뒤에 쓰려고)
                    value=tokenizer.pad_token_id,
                )
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                response_ids[:, -1] = tokenizer.pad_token_id # 맨 오른쪽에는 무조건 pad로 처리
                
                s_model_batch = {
                    "input_ids": torch.ones(args.batch_size, args.max_length, dtype=torch.long) * tokenizer.eos_token_id,
                    "attention_mask": torch.zeros(args.batch_size, args.max_length),
                }
                if args.model_type in ["gpt2"]:
                    s_model_batch["position_ids"] = torch.zeros(args.batch_size, args.max_length, dtype=torch.long)    
                s_no_model_batch = {
                    "label": torch.ones(args.batch_size, args.max_length, dtype=torch.long) * -100,
                    "loss_mask": torch.zeros(args.batch_size, args.max_length)
                }
                
                for i, input_ids_ in enumerate(model_batch['input_ids']):
                    source_len = int(model_batch['attention_mask'][i].sum() - no_model_batch['loss_mask'][i].sum()) + 1 # loss에 포함 안 되는, 입력 부분 길이
                    input_ids = np.concatenate([np.array(input_ids_[:source_len].cpu()), np.array(response_ids[i][:response_ids[i].tolist().index(tokenizer.pad_token_id) + 1].cpu())], axis=0)
                    # 데이터의 입력에서 입력 부분만 선택 + 생성된 부분 중 pad가 나오기 전까지 선택
                    input_ids = input_ids[:args.max_length] # max_length 넘어가면 자름
                    input_len = len(input_ids)
                    s_model_batch["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
                    s_model_batch["attention_mask"][i][:input_len-1] = 1.0 # 뒤에 쓸 데이터로 바꾸기 위해, 1로 처리
                    if args.model_type in ["gpt2"]:
                        s_model_batch["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
                    s_no_model_batch["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long) # student의 입출력을 label로
                    s_no_model_batch["label"][i][:source_len-1] = -100 # student의 입력 부분은 loss에 포함 안 시키기 위해
                    s_no_model_batch["loss_mask"][i][:input_len-1] = 1.0
                    s_no_model_batch["loss_mask"][i][:source_len-1] = 0 # student의 입력 부분은 loss에 포함 안 시키기 위해
            dataset["train"].move_to_device(s_model_batch, s_no_model_batch, gen_data, device)

            ### 2. Training Teacher ###
            teacher_model.train()
            outputs = teacher_model(**s_model_batch, use_cache=False)
            logits = outputs.logits[:, args.prompt_len:, :] # (batch,sequence_length,vocab_size) shape에서 prompt_len 이후만 사용

            if args.base_coef != 0.0:
                base_ratio = float(global_step) / args.total_iters
                base_distil_loss = get_distil_loss(args, tokenizer, teacher_model, teacher_model, s_model_batch, s_no_model_batch, logits, is_base=True) # 논문 상에서, L_{reg}
                teacher_distil_loss = get_distil_loss(args, tokenizer, teacher_model, model, s_model_batch, s_no_model_batch, logits, is_teacher=True) # 논문 상에서, L_{kd}
                teacher_loss = (1 - base_ratio) * args.base_coef * base_distil_loss + teacher_distil_loss # base_coef를 1로 설정해서, 논문상 식 (3)과 동일하게.
            else:
                teacher_distil_loss = get_distil_loss(args, tokenizer, teacher_model, model, s_model_batch, s_no_model_batch, logits, is_teacher=True)
                teacher_loss = teacher_distil_loss
                
            teacher_model.backward(teacher_loss) # Teacher 모델에 붙었던 Prompt Tuning 용 앞에 부분만 학습
            teacher_model.step()
            
            dist.all_reduce(teacher_loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = teacher_loss.item() / dp_world_size

            teacher_global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(teacher_distil_loss, dist.ReduceOp.SUM, group=dp_group)
                teacher_global_distil_loss = teacher_distil_loss.item() / dp_world_size
                total_distil_loss += teacher_global_distil_loss

            ### 3. Training Student ###
            model.train()
            teacher_model.eval()

            outputs = model(**s_model_batch, use_cache=False)
            logits = outputs.logits
            
            distil_loss = get_distil_loss(args, tokenizer, model, teacher_model, s_model_batch, s_no_model_batch, logits) # 일반적인 KD (reverse)
            loss = distil_loss

            model.backward(loss)
            model.step()
            
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss += loss.item() / dp_world_size

            global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
            
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        os.makedirs(os.path.join(save_dir_path, "teacher"), exist_ok=True)
                        model.module.config.to_json_file(os.path.join(save_dir_path, "config.json"))
                        teacher_model.module.config.to_json_file(os.path.join(save_dir_path, "teacher", "config.json"))
                        tokenizer.save_pretrained(save_dir_path)
                    if mpu.get_data_parallel_rank() == 0:
                        save_parallel(model.module, save_dir_path)
                        save_parallel(teacher_model.module, os.path.join(save_dir_path, "teacher"))
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        os.makedirs(os.path.join(save_dir_path, "teacher"), exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                        teacher_model.module.save_pretrained(os.path.join(save_dir_path, "teacher"), safe_serialization=False)
                dist.barrier()

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                best_eval = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, best_eval, teacher_model)
                    
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break
            
    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, best_eval=None, teacher_model=None):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
        loss_func = mpu.parallel_cross_entropy
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            # dist.barrier()
            # for rank in range(dist.get_world_size()):
            #     if dist.get_rank() == rank:
            #         print(f"rank: {dist.get_rank()}", model_batch["input_ids"][0][:128])
            #     dist.barrier()
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                lm_losses = loss_func(logits.contiguous().float(), no_model_batch["label"]).view(-1)
                loss_mask = no_model_batch["loss_mask"].view(-1)
                loss = (lm_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                ) # eval_gen이 True 일 때 굳이 padding을 해야 하는가?
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if args.model_parallel and mpu.get_data_parallel_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)
        
            if get_rank() == 0:
                eval_dir = os.path.join(args.save, "eval", str(epoch))
                print_rank(eval_dir)
                os.makedirs(eval_dir, exist_ok=True)
                with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                    for resp in responses:
                        f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
        
        if get_rank() == 0:
            avg_loss = all_loss / step
        
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
            print_rank(log_str)
            save_rank(log_str, os.path.join(args.save, "log.txt"))

        if best_eval is not None and res['rougeL'] > best_eval:
            save_dir = os.path.join(args.save, "best_rougeL")
            if get_rank() == 0:
                print_rank(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(os.path.join(save_dir, "teacher"), exist_ok=True)

                with open(os.path.join(save_dir, "log.txt"), "w") as f:
                    f.write("epoch: %s, rougeL: %f\n" % (str(epoch), res['rougeL']))

            best_eval = res['rougeL']

            if get_rank() == 0:
                model.module.config.to_json_file(os.path.join(save_dir, "config.json"))
                teacher_model.module.config.to_json_file(os.path.join(save_dir, "teacher", "config.json"))
                tokenizer.save_pretrained(save_dir)
            save_parallel(model.module, save_dir)
            save_parallel(teacher_model.module, os.path.join(save_dir, "teacher"))


    elif get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))

        if best_eval is not None and res['rougeL'] > best_eval:
            save_dir = os.path.join(args.save, "best_rougeL")
            print_rank(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, "teacher"), exist_ok=True)

            with open(os.path.join(save_dir, "log.txt"), "w") as f:
                f.write("epoch: %s, rougeL: %f\n" % (str(epoch), res['rougeL']))

            best_eval = res['rougeL']
        
            print_rank(f"Model save to {save_dir}")
            tokenizer.save_pretrained(save_dir)
            model.module.save_pretrained(save_dir, safe_serialization=False)
            teacher_model.module.save_pretrained(os.path.join(save_dir, "teacher"), safe_serialization=False)

    return best_eval


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    
    # reformulate_prompt_teacher(args, 'cpu')
    # exit()

    initialize(args)
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]    
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = mpu.get_data_parallel_world_size() if args.model_parallel else dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch * 10

        if args.abl_exposure:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)

    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model, teacher_optimizer, teacher_lr_scheduler = setup_teacher_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
        #teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
    
if __name__ == "__main__":
    main()
