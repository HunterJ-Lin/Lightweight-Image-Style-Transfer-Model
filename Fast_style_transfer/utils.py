from torchvision import transforms
import torch
import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.arr = []
        self.std = 0
        self.count = 0

    def update(self, val):
        self.arr.append(val)
        self.count += 1
        if self.count >= 100:
            self.std = np.std(self.arr, axis=0, ddof=1)
            std = self.std
            self.reset()
            return std


def train_transform(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    return transform


def style_transform(image_size=None):
    resize = [transforms.Resize(image_size)] if image_size else []
    centerCrop = [transforms.CenterCrop(image_size)] if image_size else []
    transform = transforms.Compose(resize + centerCrop + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform


def gram_matrix(y):
    b, c, h, w = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram


def denormalize(image):
    for c in range(3):
        image[:, c].mul_(std[c]).add_(mean[c])
    return image


def deprocess(image):
    image = denormalize(image)[0]
    image *= 255
    image = torch.clamp(image, 0, 255).cpu().numpy().astype(np.uint8)
    image = image.transpose(1, 2, 0)
    return image


def adjust_learning_rate(optimizer, learning_rate, batches_done, gammas, schedule):
    lr = learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (batches_done >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



