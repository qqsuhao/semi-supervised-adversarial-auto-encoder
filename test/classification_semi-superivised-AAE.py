# -*- coding:utf8 -*-
# @TIME     : 2020/11/1 13:51
# @Author   : Hao Su
# @File     : test_semi-superivised-AAE.py

'''
test contents:
classification accuracy of enc
distribution of latent variable
reconstruction error of AE
generalization ability of enc and AE
'''

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
sum_predictions = []
sum_targets = []
ae_criteria = nn.MSELoss()
reconstrction_error = 0
INDEX = global_index()
record = 0
with torch.no_grad():
    for inputs, targets in dataloader:
        INDEX.add_real_targets(list(targets.cpu().numpy()))
        i_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(-1, opt.imageSize ** 2)
        z_enc, c_enc = enc(inputs)
        dec_input = torch.cat((z_enc, c_enc), dim=1)
        outputs = dec(dec_input)

        tmp = c_enc.detach().cpu().numpy()
        INDEX.add_fake_targets(list(np.argmax(tmp, axis=1)))

        INDEX.add_letent_z(z_enc.cpu().numpy())

        loss = ae_criteria(outputs, inputs)
        reconstrction_error += loss.item() * i_size

        ## record results
        if record % opt.sample_interval == 0:
            # ae_outputs.data = ae_outputs.data.mul(0.5).add(0.5)
            vutils.save_image(outputs.view(-1, 1, opt.imageSize, opt.imageSize),
                              '{0}/ae_outputs_{1}.png'.format(opt.experiment, record))
            # inputs.data = inputs.data.mul(0.5).add(0.5)
            vutils.save_image(inputs.view(-1, 1, opt.imageSize, opt.imageSize),
                              '{0}/ae_inputs_{1}.png'.format(opt.experiment, record))

        record += 1


## end of testing
reconstrction_error = reconstrction_error / opt.dataSize
classification_error = INDEX.classification_accuracy()
INDEX.plot_latent_distribution("tsne for latent space",
                          "{0}/tsne_z.png".format(opt.experiment))
INDEX.plot_2D_scatter("tsne for latent space",
                      "{0}/2D_z.png".format(opt.experiment))
print("reconstrction_error: ", reconstrction_error)
print("classification_error: ", classification_error)