from torchstat import stat
import argparse
import model
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Test Model Information")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Optional path to checkpoint model")
    parser.add_argument("--model_name", type=str, required=True, help="TransformerNet Type")
    parser.add_argument("--use_cuda", type=bool, default=True, help="use cuda or not")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda else "cpu")
    net = model.__dict__[args.model_name]().to()
    net.load_state_dict(torch.load(args.checkpoint_model))
    stat(net, (3, 512, 512))
