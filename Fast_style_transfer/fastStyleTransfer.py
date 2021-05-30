import argparse
import os
from torchvision.utils import save_image
import model
import time
from utils import *
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--output_direction_path", type=str, default="images/outputs", help="Path to output direction")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    parser.add_argument("--model_name", type=str, default="TransformerNet_Scale_Fire", help="TransformerNet Type")
    args = parser.parse_args()

    os.makedirs(args.output_direction_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputImageTransform = style_transform()

    transformerNet = model.__dict__[args.model_name]().to(device)
    transformerNet.load_state_dict(torch.load(args.checkpoint_model))
    transformerNet.eval()

    image = inputImageTransform(Image.open(args.image_path).convert("RGB")).to(device)
    image = image.unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        stylized_image = denormalize(transformerNet(image)).cpu()
    end = time.time()
    print("time consumption：", str(end - start))
    fn = args.image_path.split("/")[-1]
    save_image(stylized_image, f"{args.output_direction_path}/stylized-{fn}")
