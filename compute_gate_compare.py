import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description='compute_gate_score')
    parser.add_argument('--gate-orig', default=None, help='path to model log.txt')
    parser.add_argument('--label-orig', default=None, help='metric type')
    parser.add_argument('--gate-sar', default=None, help='path to model log.txt')
    parser.add_argument('--label-sar', default=None, help='metric type')
    parser.add_argument('--model-size', default=None, help='model size')
    args = parser.parse_args()
    print("original gate:", args.gate_orig)
    print("original label:", args.label_orig)
    print("SAR gate:", args.gate_sar)
    print("SAR label:", args.label_sar)
    gate_files_orig = [name for name in os.listdir(args.gate_orig) if os.path.isfile(os.path.join(args.gate_orig, name))]
    label_files_orig = [name for name in os.listdir(args.label_orig) if os.path.isfile(os.path.join(args.label_orig, name))]
    gate_files_sar = [name for name in os.listdir(args.gate_sar) if os.path.isfile(os.path.join(args.gate_sar, name))]
    label_files_sar = [name for name in os.listdir(args.label_sar) if os.path.isfile(os.path.join(args.label_sar, name))]
    assert len(gate_files_orig) == len(label_files_orig)
    assert len(gate_files_sar) == len(label_files_sar)

    kl_div = nn.KLDivLoss(reduction='none')
    temp_logits = torch.load(os.path.join(args.gate_orig, "1.pt"), map_location=torch.device('cpu'))
    layer_num = len(temp_logits)
    torch.set_printoptions(precision=5)
    print(f"Layer\tmean\tmax\tmin\tmax_top_logits_orig\tmax_top_logits_sar\tmin_top_logits_orig\tmin_top_logits_sar\tnegative_kl_div\ttotal_tokens")

    for layer_idx in range(layer_num):
        layer_kl_div = []
        layer_top_logits_orig = []
        layer_top_logits_sar = []
        negative_kl_div = 0
        for idx in range(len(gate_files_orig)):
            gate_logits_orig = torch.load(os.path.join(args.gate_orig, f"{idx+1}.pt"), map_location=torch.device('cpu'))
            gate_logits_sar = torch.load(os.path.join(args.gate_sar, f"{idx+1}.pt"), map_location=torch.device('cpu'))
            label = torch.load(os.path.join(args.label_orig, f"{idx+1}.pt"), map_location=torch.device('cpu')).view(-1)

            gate_logit_orig = gate_logits_orig[layer_idx]
            gate_logit_orig = F.softmax(gate_logit_orig.to(torch.float64), dim=1)
            gate_logit_sar = gate_logits_sar[layer_idx]
            gate_logit_sar = F.softmax(gate_logit_sar.to(torch.float64), dim=1)

            #top_logits, top_indices = gate_logit.topk(gate_logit.shape[-1], dim=1)
            valid_top_logits_orig = gate_logit_orig[(label != -100).nonzero()].squeeze()  # (response_part_length, # experts)
            valid_top_logits_sar = gate_logit_sar[(label != -100).nonzero()].squeeze()

            # Compute the KL divergence between the two distributions
            kl_div_loss = kl_div(valid_top_logits_sar.log(), valid_top_logits_orig).sum(-1)
            for token_idx in range(len(kl_div_loss)):
                if kl_div_loss[token_idx] < 0:
                    negative_kl_div += 1
                    kl_div_loss[token_idx] = 0.0
            # stack valid_top_logits to layer_top_logits
            layer_kl_div.append(kl_div_loss)
            layer_top_logits_orig.append(valid_top_logits_orig)
            layer_top_logits_sar.append(valid_top_logits_sar)
          
        layer_kl_div = torch.cat(layer_kl_div, dim=0)
        layer_top_logits_orig = torch.cat(layer_top_logits_orig, dim=0)
        layer_top_logits_sar = torch.cat(layer_top_logits_sar, dim=0)

        mean_kl_loss = layer_kl_div.mean(0)
        max_kl_loss = layer_kl_div.max(0).values
        min_kl_loss = layer_kl_div.min(0).values
        # print(f"Layer {layer_idx+1} mean KL divergence: {mean_kl_loss}")
        # print(f"Layer {layer_idx+1} max KL divergence: {max_kl_loss}")
        # print(f"Layer {layer_idx+1} min KL divergence: {min_kl_loss}")
        # print(f"Layer {layer_idx+1} max top logits orig: {layer_top_logits_orig[layer_kl_div.max(0).indices]}")
        # print(f"Layer {layer_idx+1} max top logits sar: {layer_top_logits_sar[layer_kl_div.max(0).indices]}")
        # print(f"Layer {layer_idx+1} min top logits orig: {layer_top_logits_orig[layer_kl_div.min(0).indices]}")
        # print(f"Layer {layer_idx+1} min top logits sar: {layer_top_logits_sar[layer_kl_div.min(0).indices]}")
        # # print(f"Layer {layer_idx+1} min top logits orig sum : {layer_top_logits_orig[layer_kl_div.min(0).indices].sum()}")
        # # print(f"Layer {layer_idx+1} min top logits sar sum : {layer_top_logits_sar[layer_kl_div.min(0).indices].sum()}")
        # print(f"Total negative KL div count: {negative_kl_div} / Total tokens: {len(layer_kl_div)}")
        # #print("---------------------------------------------")
        print(f"{layer_idx+1}\t{mean_kl_loss}\t{max_kl_loss}\t{min_kl_loss}\t{layer_top_logits_orig[layer_kl_div.max(0).indices].tolist()}\t{layer_top_logits_sar[layer_kl_div.max(0).indices].tolist()}\t{layer_top_logits_orig[layer_kl_div.min(0).indices].tolist()}\t{layer_top_logits_sar[layer_kl_div.min(0).indices].tolist()}\t{negative_kl_div}\t{len(layer_kl_div)}")

        layer_kl_div = layer_kl_div.tolist()
        # File writing these values
        # csv file, each row is a layer, each column is a {Layer, mean, max, min, max_top_logits_orig, max_top_logits_sar, min_top_logits_orig, min_top_logits_sar, negative_kl_div, total_tokens}
        # indent each column with a tab
        with open(f"SAR_kl_div_values_{args.model_size}.csv", 'a') as f: 
            f.write(f"{layer_idx+1}")
            for token_idx in range(len(layer_kl_div)):
                f.write(f",{layer_kl_div[token_idx]:.4e}")
            f.write("\n")
        
        del layer_kl_div, layer_top_logits_orig, layer_top_logits_sar

if __name__ == "__main__":
    main()