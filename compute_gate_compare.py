import argparse
import os
import torch
import torch.nn as nn

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

    import pdb
    pdb.set_trace()
    softmax = nn.Softmax(1)
    temp_logits = torch.load(os.path.join(args.gate_orig, "1.pt"), map_location=torch.device('cpu'))
    layer_num = len(temp_logits)

    for layer_idx in range(layer_num):
        layer_top_logits = []
        for idx in range(len(gate_files)):
            gate_logits = torch.load(os.path.join(args.gate, f"{idx+1}.pt"))
            label = torch.load(os.path.join(args.label, f"{idx+1}.pt")).view(-1)

            gate_logit = gate_logits[layer_idx]
            gate_logit = softmax(gate_logit.to(torch.float32))

            top_logits, top_indices = gate_logit.topk(gate_logit.shape[-1], dim=1)
            valid_top_logits = top_logits[(label != -100).nonzero()].squeeze()  # (response_part_length, # experts)

            # stack valid_top_logits to layer_top_logits
            layer_top_logits.append(valid_top_logits)
        layer_top_logits = torch.cat(layer_top_logits, dim=0)
        layer_top_logits = layer_top_logits.mean(0)
        print(f"Layer {layer_idx+1} mean top logits: {layer_top_logits}")

if __name__ == "__main__":
    main()