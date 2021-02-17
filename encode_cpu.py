import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

from model import Generator, Encoder
from train_encoder import VGGLoss
from src.loss.loss_provider import LossProvider

import matplotlib.pyplot as plt


def image2tensor(image):
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1, 2, 0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()


device = 'cpu'
image_size = 256

g_model_path = '/media/morzh/ext4_volume/work/stylegan2-pytorch/models/generator_ffhq.pt'
g_ckpt = torch.load(g_model_path, map_location=device)

latent_dim = g_ckpt['args'].latent
# latent_dim = 14

generator = Generator(image_size, latent_dim, 8).to(device)
generator.load_state_dict(g_ckpt["g_ema"], strict=False)
generator.eval()
print('[generator loaded]')

e_model_path = '/media/morzh/ext4_volume/work/stylegan2-pytorch/models/encoder_ffhq.pt'
e_ckpt = torch.load(e_model_path, map_location=device)

encoder = Encoder(image_size, latent_dim).to(device)
encoder.load_state_dict(e_ckpt['e'])
encoder.eval()
print('[encoder loaded]')

truncation = 0.7
trunc = generator.mean_latent(4096).detach().clone()


import pathlib

batch_size = 1

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

root_dir = os.path.join(pathlib.Path().absolute(), 'examples/Dataset')
print(root_dir)

dataset = datasets.ImageFolder(root=root_dir, transform=transform)
loader = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))

imgs, _ = next(loader)
imgs = imgs.to(device)

with torch.no_grad():
    z0 = encoder(imgs)
    imgs_gen, _ =  generator([z0], input_is_latent=True, truncation=truncation, truncation_latent=trunc, randomize_noise=False)

imgs_real = torch.cat([img for img in imgs], dim=1)
imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)

print('initial projections:')
imshow(tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)), 10)

import time

vgg_loss = VGGLoss(device)
provider = LossProvider()
loss_function = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')

z = z0.detach().clone()
z.requires_grad = True

optimizer = torch.optim.Adam([z], lr=0.01)

start_time = time.time()
for step in range(1101):
    imgs_gen, _ = generator([z], input_is_latent=True, truncation=truncation, truncation_latent=trunc, randomize_noise=False)
    z_hat = encoder(imgs_gen)

    loss_watson = F.mse_loss(imgs_gen, imgs) + loss_function(imgs_gen, imgs) + F.mse_loss(z0, z_hat) * 2.0
    loss_vgg = F.mse_loss(imgs_gen, imgs) + vgg_loss(imgs_gen, imgs) + F.mse_loss(z0, z_hat) * 2.0

    optimizer.zero_grad()
    loss_vgg.backward()
    optimizer.step()

    if (step + 1) % 100 == 0:
        print(f'step:{step + 1}, loss:{loss_vgg.item()}')
        # imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)
        # imshow(tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)),10)

end_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))
print(z.shape)

imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)
imshow(tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)), 10)

