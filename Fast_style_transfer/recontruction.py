import argparse
from model import *
import torch
import numpy as np
import time
from torchstat import stat
from torchvision.utils import save_image
from utils import *
from PIL import Image


def get_small_model(large_model, model_name):
    small_model = extract_para(large_model, model_name)
    return small_model


def extract_para(large_model_state_dict, model_name):
    if "Scale" in model_name:
        item = list(large_model_state_dict.items())
        dict = {}
        for i in range(len(item)):
            if ("mask" in item[i][0]):
                dict[item[i][0]] = item[i][1].item()
                # if item[i][1].item() == 0:
                #     print(item[i][0])
        num_block = sum(np.array(list(dict.values())) != 0)
        print(num_block)
        if "Fire" in model_name:
            small_net = TransformerNet_Fire_Small(num_block)
        else:
            small_net = TransformerNet_Small(num_block)

        small_net_state_dict = small_net.state_dict()
        reversed_dict = {k: v for k, v in large_model_state_dict.items()}

        arr = []
        cnt = 0
        for i in range(len(item)):
            if ("mask" in item[i][0]):
                k = item[i][0]
                v = item[i][1]
                if v != 0:
                    for j in range(i - 8, i):
                        tmp = item[j][0].split('.')
                        tmp[2] = str(cnt)
                        string = ''
                        for s in tmp:
                            string += s + '.'
                        string = string[:-1]
                        # print(string)
                        arr.append((string, item[j][1]))
                    cnt += 1
                for j in range(i - 8, i + 1):
                    del reversed_dict[item[j][0]]

        for t in arr:
            reversed_dict[t[0]] = t[1]

        small_net_state_dict.update(reversed_dict)
        small_net.load_state_dict(small_net_state_dict)
        return small_net


class TransformerNet_Small(nn.Module):
    def __init__(self, num_block):
        super(TransformerNet_Small, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 64, kernel_size=3, stride=1),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ConvBlock(128, 128, kernel_size=3, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=2),
            ConvBlock(256, 256, kernel_size=3, stride=1),
            make_layer(ResidualBlock, num_block=num_block, in_channels=256, out_channels=256, kernel_size=3, stride=1,
                       upsample=False, normalize=True, relu=True),
            ConvBlock(256, 256, kernel_size=3, upsample=False),
            ConvBlock(256, 128, kernel_size=3, upsample=True),
            ConvBlock(128, 128, kernel_size=3, upsample=False),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 64, kernel_size=3, upsample=False),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, upsample=False, normalize=False, relu=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class TransformerNet_Fire_Small(nn.Module):
    def __init__(self, num_block):
        super(TransformerNet_Fire_Small, self).__init__()
        self.model = nn.Sequential(
            ConvBlock_Fire(3, 32, kernel_size=9, stride=1),
            Fire(inplanes=32, squeeze_planes=8, expand1x1_planes=32, expand3x3_planes=32, stride=2),
            Fire(inplanes=64, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64, stride=2),
            Fire(inplanes=128, squeeze_planes=32, expand1x1_planes=128, expand3x3_planes=128, stride=2),
            make_layer(ResidualBlock, num_block=num_block, in_channels=256, out_channels=256, kernel_size=3, stride=1,
                       upsample=False, normalize=True, relu=True),
            Fire(inplanes=256, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64, upsample=True),
            Fire(inplanes=128, squeeze_planes=8, expand1x1_planes=32, expand3x3_planes=32, upsample=True),
            Fire(inplanes=64, squeeze_planes=4, expand1x1_planes=16, expand3x3_planes=16, upsample=True),
            ConvBlock_Fire(32, 3, kernel_size=9, upsample=False, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", type=str, default="c:/users/hunterj/desktop/jd.jpg", help="Path to image")
    # parser.add_argument("--output_direction_path", type=str, default="images/outputs", help="Path to output direction")
    # args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # large_model_state_dict = torch.load("c:/users/hunterj/desktop/feathers_Scale/26000.pth")
    # #
    # # stat(TransformerNet(), (3, 256, 256))
    # # stat(get_small_model(large_model_state_dict, "TransformerNet_Scale"), (3, 256, 256))
    # # large_model_state_dict = torch.load("c:/users/hunterj/desktop/starry_night_scale_fire/42000.pth")
    #
    # # stat(TransformerNet_Fire(), (3, 256, 256))
    # # stat(get_small_model(large_model_state_dict, "TransformerNet_Scale_Fire"), (3, 256, 256))
    # small_model = get_small_model(large_model_state_dict, "TransformerNet_Scale").to(device)
    # small_model.eval()
    #
    # inputImageTransform = style_transform()
    #
    # image = inputImageTransform(Image.open(args.image_path).convert("RGB")).to(device)
    # image = image.unsqueeze(0)
    #
    # with torch.no_grad():
    #     stylized_image = denormalize(small_model(image)).cpu()
    #
    # fn = args.image_path.split("/")[-1]
    # save_image(stylized_image, f"{args.output_direction_path}/stylized-{fn}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    parser.add_argument("--model_name", type=str, default="TransformerNet_Scale_Fire", help="TransformerNet Type")
    args = parser.parse_args()
    large_model_state_dict = torch.load(args.checkpoint_model)
    small_model = get_small_model(large_model_state_dict, args.model_name)
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
