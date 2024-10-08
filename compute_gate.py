import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser(description='compute_gate_score')
    parser.add_argument('--gate', default=None, help='path to model log.txt')
    parser.add_argument('--label', default=None, help='metric type')
    args = parser.parse_args()

    gate_files = [name for name in os.listdir(args.gate) if os.path.isfile(os.path.join(args.gate, name))]
    label_files = [name for name in os.listdir(args.label) if os.path.isfile(os.path.join(args.label, name))]
    assert len(gate_files) == len(label_files)

    for idx in range(len(gate_files)):
        gate_logits = torch.load(os.path.join(args.gate, f"{idx}.pt"))
        label = torch.load(os.path.join(args.label, f"{idx}.pt"))

        import pdb
        pdb.set_trace()

        (no_model_batch["label"] != -100).int()

if __name__ == "__main__":
    main()