import argparse
import os
import random
import glob
import time
from recontruction import get_small_model
from torchvision import datasets
from torch.utils.data import DataLoader
from fista import FISTA
from utils import *
import model
from model import Mask_FPGM
from torch.optim import Adam
import torch
from torchvision.utils import save_image
from PIL import Image
from recontruction import TransformerNet_Fire_Small, TransformerNet_Small

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Fast-Neural-Style")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="path to training dataset")
    parser.add_argument("--style_image", type=str, default="images/styles/cubist.jpg", help="path to style image")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=256, help="Size of training images")
    parser.add_argument("--style_size", type=int, help="Size of style image")
    parser.add_argument("--lambda_content", type=float, default=1e5, help="Weight for content loss")
    parser.add_argument("--lambda_style", type=float, default=1e10, help="Weight for style loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sparse_lambda", type=float, default=1, help="sparse lambda")
    parser.add_argument("--mask_step", type=float, default=100, help="mask step")
    parser.add_argument("--gammas", type=list, default=[0.1, 0.1, 0.1],
                        help="sets the learning rate to the initial LR decayed by them")
    parser.add_argument("--schedule", type=list,
                        default=[40000, 80000, 140000],
                        help="the schedule to change learning rate")
    parser.add_argument("--checkpoint_model", type=str, help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Batches between saving model")
    parser.add_argument("--sample_interval", type=int, default=100, help="Batches between saving image samples")
    parser.add_argument("--model_name", type=str, default="TransformerNet", help="TransformerNet Type")
    parser.add_argument("--vgg_type", type=str, default="VGG16", help="VGG Type")
    parser.add_argument('--dist_type', default='l2', type=str, choices=['l2', 'l1', 'cos'], help='distance type of GM')
    parser.add_argument('--rate_norm', type=float, default=0.9, help='the remaining ratio of pruning based on Norm')
    parser.add_argument('--rate_dist', type=float, default=0.1, help='the reducing ratio of pruning based on Distance')
    parser.add_argument('--layer_begin', type=int, default=0, help='compress layer of model')
    parser.add_argument('--layer_end', type=int, default=184, help='compress layer of model')
    args = parser.parse_args()

    style_name = args.style_image.split("/")[-1].split(".")[0]
    time_str = time.strftime('%Y-%m-%d+%H_%M_%S', time.localtime(time.time()))
    os.makedirs(f"images/outputs/{style_name}-training-{time_str}", exist_ok=True)
    os.makedirs(f"checkpoints/{style_name}-{time_str}", exist_ok=True)
    config = open(f"checkpoints/{style_name}-{time_str}/config.txt", "w+")
    stylelossfile = open(f"checkpoints/{style_name}-{time_str}/styleloss.txt", "w+")
    contentlossfile = open(f"checkpoints/{style_name}-{time_str}/contentloss.txt", "w+")
    totallossfile = open(f"checkpoints/{style_name}-{time_str}/totalloss.txt", "w+")
    style_meter = AverageMeter()
    content_meter = AverageMeter()
    total_meter = AverageMeter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.ImageFolder(args.dataset_path, train_transform(args.image_size))
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformerNet = TransformerNet_Fire_Small(7).to(device)
    vgg = model.__dict__[args.vgg_type](require_grad=False).to(device)
    m = Mask_FPGM(transformerNet)
    m.init_length()
    m.model = transformerNet
    m.init_mask(args.rate_norm, args.rate_dist, args.dist_type, layer_begin=args.layer_begin, layer_end=args.layer_end)
    m.do_mask()
    m.do_similar_mask()
    transformerNet = m.model

    for (key, value) in zip(vars(args).keys(), vars(args).values()):
        config.write(f"{key}:{value}")
        config.write('\n')
    config.write(str(vgg.weights))
    config.write('\n')
    config.write(str(vgg))
    config.write('\n')
    config.write(str(transformerNet))
    config.close()

    if args.checkpoint_model:
        transformerNet.load_state_dict(torch.load(args.checkpoint_model))

    current_learning_rate = args.lr
    optimizer_s = Adam(transformerNet.parameters(), lr=current_learning_rate)
    l2_loss = torch.nn.MSELoss().to(device)

    # load style image
    style = style_transform(args.style_size)(Image.open(args.style_image).convert("RGB"))
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # extrace style features
    feature_style, _ = vgg(style)
    gram_style = [gram_matrix(x) for x in feature_style]

    # sample 8 images for visual evaluation of the model
    image_samples = []
    for path in random.sample(glob.glob(f"{args.dataset_path}/*/*.jpg"), 8):
        item = style_transform(args.image_size)(Image.open(path).convert("RGB"))
        # print(item.shape)
        image_samples += [item]

    image_samples = torch.stack(image_samples)


    def save_sample(batches_done):
        """ Evaluates the model and saves image samples """
        transformerNet.eval()
        # print(image_samples.cpu().shape)
        with torch.no_grad():
            output = transformerNet(image_samples.to(device))
        image_grid = denormalize(torch.cat((image_samples.cpu(), output.cpu()), 2))
        save_image(image_grid,
                   f"images/outputs/{style_name}-training-{time_str}/{batches_done}.jpg",
                   nrow=4)
        transformerNet.train()


    for epoch in range(args.epochs):
        epoch_metrics = {"content": [], "style": [], "total": []}
        for batch_i, (images, _) in enumerate(dataloader):
            optimizer_s.zero_grad()
            images_original = images.to(device)
            # print(images_original.shape)
            images_transformed = transformerNet(images_original)

            feature_original, content_relu_original = vgg(images_original)
            feature_transformed, content_relu_transformed = vgg(images_transformed)

            # Compute content loss as MSE between features
            content_loss = args.lambda_content * l2_loss(content_relu_original, content_relu_transformed)

            # Compute style loss as MSE between gram matrices
            style_loss = 0
            for ft_y, gm_s in zip(feature_transformed, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += l2_loss(gm_y, gm_s[:images.size(0), :, :])
            style_loss *= args.lambda_style

            total_loss = content_loss + style_loss
            if torch.isnan(total_loss).item():
                continue
            total_loss.backward()
            # Mask grad for iteration
            m.do_grad_mask()
            optimizer_s.step()
            if batch_i % args.mask_step == 0:
                m.model = transformerNet
                # m.if_zero()
                m.init_mask(args.rate_norm, args.rate_dist, args.dist_type, layer_begin=args.layer_begin, layer_end=args.layer_end)
                m.do_mask()
                m.do_similar_mask()
                # m.if_zero()
                transformerNet = m.model
                transformerNet = transformerNet.cuda()

            epoch_metrics["content"] += [content_loss.item()]
            epoch_metrics["style"] += [style_loss.item()]
            epoch_metrics["total"] += [total_loss.item()]
            y1 = style_meter.update(style_loss.item())
            y2 = content_meter.update(content_loss.item())
            y3 = total_meter.update(total_loss.item())
            if y1 is not None:
                stylelossfile.write(f"{y1}\n")
                contentlossfile.write(f"{y2}\n")
                totallossfile.write(f"{y3}\n")

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
                % (
                    epoch + 1,
                    args.epochs,
                    batch_i,
                    len(dataloader),
                    content_loss.item(),
                    np.mean(epoch_metrics["content"]),
                    style_loss.item(),
                    np.mean(epoch_metrics["style"]),
                    total_loss.item(),
                    np.mean(epoch_metrics["total"]),
                )
            )

            batches_done = epoch * len(dataloader) + batch_i + 1
            # print(f"lr:{adjust_learning_rate(optimizer_s, args.lr, batches_done, args.gammas, args.schedule)}")
            if batches_done % args.sample_interval == 0:
                save_sample(batches_done)

            if args.checkpoint_interval > 0 and batches_done % args.checkpoint_interval == 0:
                style_name = os.path.basename(args.style_image).split(".")[0]
                torch.save(transformerNet.state_dict(),
                           f"checkpoints/{style_name}-{time_str}/{batches_done}.pth")

stylelossfile.close()
contentlossfile.close()
totallossfile.close()
config.close()
