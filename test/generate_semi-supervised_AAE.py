# -*- coding:utf8 -*-
# @TIME     : 2020/11/21 16:48
# @Author   : SuHao
# @File     : generate_semi-supervised_AAE.py


import os
import tqdm
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
from test.model_test import global_index
import torchvision.transforms as transforms
from models.MLP_AAE import Decoder, Encoder
from dataload.dataload import load_dataset
from dataload.self_transforms import AddGaussianNoise, AddSaltPepperNoise, Fllip


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/semi-supervised-AAE-mnist_test", help="path to save experiments results")
parser.add_argument("--dataset", default="mnist", help="mnist")
parser.add_argument('--dataroot', default=r"../data", help='path to dataset')
parser.add_argument("--enc_pth", default=r"../experiments/semi-supervised-AAE-mnist/enc.pth", help="pretrained model of encoder")
parser.add_argument("--dec_pth", default=r"../experiments/semi-supervised-AAE-mnist/dec.pth", help="pretrained model of decoder")
parser.add_argument("--batchSize", type=int, default=128, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--nz", type=int, default=10, help="dimensionality of the latent space")
parser.add_argument("--imageSize", type=int, default=28, help="size of each image dimension")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## load dataset
os.makedirs(os.path.join(opt.dataroot, opt.dataset), exist_ok=True)
trans = transforms.Compose([transforms.ToTensor()])
dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans, train=False)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
opt.dataSize = len(dataset)

## parameters
opt.n_classes = len(dataset.classes)

## load pretrained model
enc = Encoder(opt.imageSize**2, opt.nz, opt.n_classes).to(device)
dec = Decoder(opt.imageSize**2, opt.nz, opt.n_classes).to(device)
enc.load_state_dict(torch.load(opt.enc_pth))
dec.load_state_dict(torch.load(opt.dec_pth))
print("Pretrained models have been loaded.")


## testing
enc.eval()
dec.eval()

## Gaussian Distribution
def gen_z_gauss(i_size, nz):
    return torch.randn(i_size, nz).to(device)

def gen_z_cat(i_size):
    random_targets = torch.randint(0, 10, (i_size, 1)).to(device)
    one_hot = torch.zeros((i_size, opt.n_classes), device=device)
    return one_hot.scatter(1, random_targets.view(-1, 1), 1)

## generate
z_enc = gen_z_gauss(opt.batchSize, opt.nz)
c_enc = gen_z_cat(opt.batchSize)
with torch.no_grad():
    dec_input = torch.cat((z_enc, c_enc), dim=1)
    outputs = dec(dec_input)
    vutils.save_image(outputs.view(-1, 1, opt.imageSize, opt.imageSize),
                      '{0}/ae_outputs.png'.format(opt.experiment))
