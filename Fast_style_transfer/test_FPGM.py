import argparse
import model
from model import *
import torch
import numpy as np
import time
from torchstat import stat
from torchvision.utils import save_image
from utils import *
from PIL import Image




class TransformerNet_Small(nn.Module):
    def __init__(self, num_block):
        super(TransformerNet_Small, self).__init__()
        # SFP修剪率10%，FPGM修剪率10%
        self.model = nn.Sequential(
            ConvBlock(3, 32-3-3, kernel_size=9, stride=1),
            ConvBlock(32-3-3, 64-6-6, kernel_size=3, stride=2),
            ConvBlock(64-6-6, 64-6-6, kernel_size=3, stride=1),
            ConvBlock(64-6-6, 128-12-12, kernel_size=3, stride=2),
            ConvBlock(128-12-12, 128-12-12, kernel_size=3, stride=1),
            ConvBlock(128-12-12, 256-25-25, kernel_size=3, stride=2),
            ConvBlock(256-25-25, 256-25-25, kernel_size=3, stride=1),
            make_layer(ResidualBlock, num_block=num_block, in_channels=256-25-25, out_channels=256-25-25, kernel_size=3, stride=1,
                       upsample=False, normalize=True, relu=True),
            ConvBlock(256-25-25, 256-25-25, kernel_size=3, upsample=False),
            ConvBlock(256-25-25, 128-12-12, kernel_size=3, upsample=True),
            ConvBlock(128-12-12, 128-12-12, kernel_size=3, upsample=False),
            ConvBlock(128-12-12, 64-6-6, kernel_size=3, upsample=True),
            ConvBlock(64-6-6, 64-6-6, kernel_size=3, upsample=False),
            ConvBlock(64-6-6, 32-3-3, kernel_size=3, upsample=True),
            ConvBlock(32-3-3, 3, kernel_size=9, upsample=False, normalize=False, relu=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class TransformerNet_Fire_Small(nn.Module):
    def __init__(self, num_block):
        super(TransformerNet_Fire_Small, self).__init__()
        self.model = nn.Sequential(
            ConvBlock_Fire(3, 32-3-3, kernel_size=9, stride=1),
            Fire(inplanes=32-3-3, squeeze_planes=8, expand1x1_planes=32-6, expand3x3_planes=32-6, stride=2),
            Fire(inplanes=64-6-6, squeeze_planes=16, expand1x1_planes=64-12, expand3x3_planes=64-12, stride=2),
            Fire(inplanes=128-12-12, squeeze_planes=32, expand1x1_planes=128-25, expand3x3_planes=128-25, stride=2),
            make_layer(ResidualBlock, num_block=num_block, in_channels=256-25-25, out_channels=256-25-25, kernel_size=3, stride=1,
                       upsample=False, normalize=True, relu=True),
            Fire(inplanes=256-25-25, squeeze_planes=16, expand1x1_planes=64-12, expand3x3_planes=64-12, upsample=True),
            Fire(inplanes=128-12-12, squeeze_planes=8, expand1x1_planes=32-6, expand3x3_planes=32-6, upsample=True),
            Fire(inplanes=64-6-6, squeeze_planes=4, expand1x1_planes=16-3, expand3x3_planes=16-3, upsample=True),
            ConvBlock_Fire(32-3-3, 3, kernel_size=9, upsample=False, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="images/content/jd.jpg", help="Path to image")
    args = parser.parse_args()
    small_model = TransformerNet_Fire_Small(7)
    inputImageTransform = style_transform()
    device = torch.device("cuda")
    image = inputImageTransform(Image.open(args.image_path).convert("RGB")).to(device)
    image = image.unsqueeze(0)
    small_model = small_model.to(device)

    start = time.time()
    with torch.no_grad():
        stylized_image = denormalize(small_model(image)).cpu()
    end = time.time()
    print("gpu time consumption：", str(end - start))
    device = torch.device("cpu")
    image = image.to(device)
    small_model = small_model.to(device)
    start = time.time()
    with torch.no_grad():
        stylized_image = denormalize(small_model(image)).cpu()
    end = time.time()
    print("cpu time consumption：", str(end - start))
    stat(small_model, (3, 512, 512))
    torch.save(small_model.state_dict(), "1.pth")
