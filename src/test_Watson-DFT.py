from loss.loss_provider import LossProvider

provider = LossProvider()
loss_function = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')

import torch
from torchvision import datasets, transforms
from PIL import Image
import os

img0 = torch.zeros(1, 3, 512, 512)
img1 = torch.zeros(1, 3, 512, 512)


path = '/media/morzh/ext4_volume/data/Faces/BeautifyMeFaceset-005/01_Neutral'
file_1 = '000002.png'
file_2 = '000003.png'

img_1 = Image.open(os.path.join(path, file_1))
img_2 = Image.open(os.path.join(path, file_2))

img_1 = transforms.ToTensor()(img_1).unsqueeze_(0)
img_2 = transforms.ToTensor()(img_2).unsqueeze_(0)

loss = loss_function(img_1, img_1)

print(loss)