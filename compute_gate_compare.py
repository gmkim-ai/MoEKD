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

    #kl_div = nn.KLDivLoss(reduction='none')
    temp_logits = torch.load(os.path.join(args.gate_orig, "1.pt"), map_location=torch.device('cpu'))
    layer_num = len(temp_logits)

    for layer_idx in range(layer_num):
        layer_top_logits_orig = []
        layer_top_logits_sar = []
        for idx in range(len(gate_files_orig)):
            gate_logits_orig = torch.load(os.path.join(args.gate_orig, f"{idx+1}.pt"), map_location=torch.device('cpu'))
            gate_logits_sar = torch.load(os.path.join(args.gate_sar, f"{idx+1}.pt"), map_location=torch.device('cpu'))
            label = torch.load(os.path.join(args.label_orig, f"{idx+1}.pt"), map_location=torch.device('cpu')).view(-1)

            gate_logit_orig = gate_logits_orig[layer_idx]
            gate_logit_orig = F.softmax(gate_logit_orig.to(torch.float32), dim=1)
            gate_logit_sar = gate_logits_sar[layer_idx]
            gate_logit_sar = F.softmax(gate_logit_sar.to(torch.float32), dim=1)

            #top_logits, top_indices = gate_logit.topk(gate_logit.shape[-1], dim=1)
            valid_top_logits_orig = gate_logit_orig[(label != -100).nonzero()].squeeze()  # (response_part_length, # experts)
            valid_top_logits_sar = gate_logit_sar[(label != -100).nonzero()].squeeze()

            

            import pdb
            pdb.set_trace()

            # stack valid_top_logits to layer_top_logits
            layer_top_logits_orig.append(valid_top_logits_orig)
            layer_top_logits_sar.append(valid_top_logits_sar)
        layer_top_logits_orig = torch.cat(layer_top_logits_orig, dim=0)
        layer_top_logits_sar = torch.cat(layer_top_logits_sar, dim=0)

        # Compute the KL divergence between the two distributions
        
        kl_loss = kl_div(layer_top_logits_sar, layer_top_logits_orig)
        print(f"Layer {layer_idx+1} KL divergence: {kl_loss}")

if __name__ == "__main__":
    main()