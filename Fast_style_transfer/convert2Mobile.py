import argparse
import os
from torchvision.utils import save_image
import model
import torch
from utils import *
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    parser.add_argument("--model_name", type=str, default="TransformerNet_Scale_Fire", help="TransformerNet Type")
    args = parser.parse_args()

    device = torch.device("cpu")

    transformerNet = model.__dict__[args.model_name]().to(device)
    transformerNet.load_state_dict(torch.load(args.checkpoint_model))
    transformerNet.eval()

    input_tensor = torch.rand(1, 3, 256, 256)

    script_model = torch.jit.trace(transformerNet, input_tensor)
    script_model.save("transformerNet.pt")
