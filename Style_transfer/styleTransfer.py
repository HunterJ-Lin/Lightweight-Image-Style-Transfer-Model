import torch.nn as nn
import torch.optim as optim
import PIL.Image as Image
import torchvision
import time
from model import get_style_model_and_loss
from torchvision.utils import save_image
from torch.optim import Adam
from torch.autograd import Variable

img_size = 512


def style_transfer(content_img, style_img, input_img, num_epoches=300):
    print('Building the style transfer model..')
    model, style_loss_list, content_loss_list = get_style_model_and_loss(
        style_img, content_img)
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])

    print('Opimizing...')
    epoch = [0]
    start = time.time()
    while epoch[0] < num_epoches:

        def closure():
            input_param.data.clamp_(0, 1)

            model(input_param)
            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

        input_param.data.clamp_(0, 1)
    end = time.time()
    print("time consumption：", str(end - start))
    return input_param.data


def load_img(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = torchvision.transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    # return img.cuda()
    return img


def show_img(img):
    img = img.squeeze(0)
    img = torchvision.transforms.ToPILImage()(img)
    img.show()


if __name__ == '__main__':
    style_img = load_img('../Fast_style_transfer/images/styles/starry_night.jpg')
    content_img = load_img('../Fast_style_transfer/images/content/dfmz.jpg')
    input_img = content_img.clone()

    out = style_transfer(content_img, style_img, input_img, num_epoches=100)

    save_image(out.cpu(), "1.jpg")
