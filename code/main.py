import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="dataset_hist", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=5, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=512, help='size of image height')
parser.add_argument('--img_width', type=int, default=512, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval between model checkpoints')
parser.add_argument('--path', type=str, default='/home/baba/EchoNus_project/code/', help='path to code and data')

opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_model/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height//2**5, opt.img_width//2**5)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = PixelDiscriminator()

if cuda:
    generator = torch.nn.DataParallel(generator).cuda()
    discriminator = torch.nn.DataParallel(discriminator).cuda()

    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(os.path.join(opt.path,'/saved_model/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch))))
    discriminator.load_state_dict(torch.load(os.path.join(opt.path,'/saved_model/%s/discriminator_%d.pth' % (opt.dataset_name, opt.epoch))))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders and Data Augmentation
transforms_ = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.RandomResizedCrop(512),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

transforms_val = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

dataloader = DataLoader(ImageDataset(os.path.join(opt.path, "Data/%s" % opt.dataset_name), transforms_=transforms_),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

print('len of train batch is: ', len(dataloader))
val_dataloader = DataLoader(ImageDataset(os.path.join(opt.path, "Data/%s" % opt.dataset_name), transforms_=transforms_val, mode='val'),
                            batch_size=1, shuffle=True, num_workers=1)
print('len of val batch is: ', len(val_dataloader))
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    real_B = Variable(imgs['B'].type(Tensor))
    fake_B = generator(real_A)
    save_image(real_A, 'images/%s/img/%s_image.png' % (opt.dataset_name, batches_done), nrow=3, normalize=True, scale_each=True)
    save_image(real_B, 'images/%s/gt/%s_image.png' % (opt.dataset_name, batches_done), nrow=3, normalize=True, scale_each=True)
    save_image(fake_B, 'images/%s/pred/%s_image.png' % (opt.dataset_name, batches_done), nrow=3, normalize=True, scale_each=True)

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(real_A,fake_B)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Loss weight of L1 pixel-wise loss between translated image and real image
        lambda_pixel = 0.99

        # Total loss
        loss_G = (1-lambda_pixel)*loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_A,real_B)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(real_A, fake_B.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_pixel.item(), loss_GAN.item(),
                                                        time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(opt.path, 'saved_model/%s/generator_%d.pth' % (opt.dataset_name, epoch)))
        torch.save(discriminator.state_dict(), os.path.join(opt.path, 'saved_model/%s/discriminator_%d.pth' % (opt.dataset_name, epoch)))
