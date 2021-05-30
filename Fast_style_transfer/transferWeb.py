import sys
from torchvision.utils import save_image
import model
import time
from utils import *
from PIL import Image


def transfer(style, method, image_t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputImageTransform = style_transform()
    print("init")
    if method == "normal":
        transformerNet = model.__dict__["TransformerNet"]().to(device)
        transformerNet.load_state_dict(torch.load(f"c:/users/hunterj/desktop/实验数据/{style}/checkpoint.pth"))
    elif method == "fire":
        transformerNet = model.__dict__["TransformerNet_Fire"]().to(device)
        transformerNet.load_state_dict(torch.load(f"c:/users/hunterj/desktop/实验数据/Fire/{style}/checkpoint.pth"))
    elif method == "scale":
        transformerNet = model.__dict__["TransformerNet_Scale"]().to(device)
        transformerNet.load_state_dict(torch.load(f"c:/users/hunterj/desktop/实验数据/Scale/{style}/checkpoint.pth"))
    elif method == "complex":
        transformerNet = model.__dict__["TransformerNet_Complex"]().to(device)
        transformerNet.load_state_dict(torch.load(f"c:/users/hunterj/desktop/实验数据/Complex/{style}/checkpoint.pth"))

    transformerNet.eval()
    image = inputImageTransform(Image.open(image_t).convert("RGB")).to(device)
    image = image.unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        stylized_image = denormalize(transformerNet(image)).cpu()
    end = time.time()
    print("time consumption:", str(end - start))
    fn = image_t.split("/")[-1]
    save_image(stylized_image, f"D:/GraduationWeb/GraduationWeb/wwwroot/stylized/stylized-{fn}")
    print(f"result:/stylized/stylized-{fn}")
    print("end")


if __name__ == '__main__':
    transfer(sys.argv[1], sys.argv[2], sys.argv[3])
