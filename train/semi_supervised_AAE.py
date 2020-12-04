# -*- coding:utf8 -*-
# @TIME     : 2020/10/30 14:42
# @Author   : Hao Su
# @File     : semi_supervised_AAE.py

'''
reference:  https://github.com/andreandradecosta/pytorch_aae
'''

from __future__ import print_function
import os
import tqdm
import torch
import argparse
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from test.model_test import global_index
from dataload.dataload import load_dataset
import torchvision.transforms as transforms
from visualization.visual import plot_tsne, plot_2D_scatter
from models.MLP_AAE import Encoder, Decoder, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/semi-supervised-AAE-mnist", help="path to save experiments results")
parser.add_argument("--dataset", default="mnist", help="mnist")
parser.add_argument('--dataroot', default=r"../data", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=20, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--nz", type=int, default=10, help="dimensionality of the latent space")
parser.add_argument("--imageSize", type=int, default=28, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)


## random seed
opt.seed = 42
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## dataset
os.makedirs(os.path.join(opt.dataroot, opt.dataset), exist_ok=True)
trans = transforms.Compose([transforms.ToTensor()])
dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans, train=True)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
opt.dataSize = len(dataset)

## parameters
opt.n_classes = len(dataset.classes)

## model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

## model
enc = Encoder(opt.imageSize**2, opt.nz, opt.n_classes).to(device)
dec = Decoder(opt.imageSize**2, opt.nz, opt.n_classes).to(device)
disc_gauss = Discriminator(opt.nz).to(device)
disc_cat = Discriminator(opt.n_classes).to(device)

enc.apply(weights_init)
dec.apply(weights_init)
disc_gauss.apply(weights_init)
disc_cat.apply(weights_init)

## reconstruction phase
ae_optimizer = optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
ae_criteria = nn.MSELoss()

# regularization phase
disc_gauss_optimizer = optim.Adam(disc_gauss.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
enc_gauss_optimizer = optim.Adam(enc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

disc_cat_optimizer = optim.Adam(disc_cat.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
enc_cat_optimizer = optim.Adam(enc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

disc_criteria = nn.BCELoss()
enc_criteria = nn.BCELoss()

# semi supervise phase
enc_classifier_optimizer = optim.Adam(enc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
enc_classifier_criteria = nn.CrossEntropyLoss()


## record results
writer = SummaryWriter("../runs{0}".format(opt.experiment[1:]), comment=opt.experiment[1:])


## Gaussian Distribution
def gen_z_gauss(i_size, nz):
    return torch.randn(i_size, nz).to(device)

def gen_z_cat(i_size):
    random_targets = torch.randint(0, 10, (i_size, 1)).to(device)
    one_hot = torch.zeros((i_size, opt.n_classes), device=device)
    return one_hot.scatter(1, random_targets.view(-1, 1), 1)


## Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t:
    for e in t:
        t.set_description(f"Epoch {e+1}/{opt.n_epoches}")

        ae_epoch_loss = 0.0
        disc_gauss_epoch_loss = 0.0
        enc_gauss_epoch_loss = 0.0
        disc_cat_epoch_loss = 0.0
        enc_cat_epoch_loss = 0.0

        INDEX = global_index()

        for inputs, targets in dataloader:
            INDEX.add_real_targets(list(targets.cpu().numpy()))
            i_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(-1, opt.imageSize**2)
            ########################
            # Reconstruction phase #
            ########################
            ae_optimizer.zero_grad()
            z_enc, c_enc = enc(inputs)
            dec_input = torch.cat((z_enc, c_enc), dim=1)
            ae_outputs = dec(dec_input)
            ae_loss = ae_criteria(ae_outputs, inputs)
            ae_loss.backward()
            ae_optimizer.step()
            ae_epoch_loss += ae_loss.item() * i_size

            # INDEX.add_letent_z(z_enc.detach().cpu().numpy())

            ########################
            # Regularization phase #
            ########################
            disc_targets = torch.cat((torch.ones(i_size, 1), torch.zeros(i_size, 1)), dim=0).to(device)
            enc_targets = torch.ones(i_size, 1).to(device)

            ## Discriminator - gauss
            z_fake, _ = enc(inputs)
            z_real = gen_z_gauss(i_size, opt.nz)
            disc_gauss_inputs = torch.cat((z_real, z_fake.detach()), dim=0)
            disc_gauss_targets = disc_targets.detach()

            disc_gauss_optimizer.zero_grad()
            disc_gauss_outputs = disc_gauss(disc_gauss_inputs)
            disc_gauss_loss = disc_criteria(disc_gauss_outputs, disc_gauss_targets)
            disc_gauss_loss.backward()
            disc_gauss_optimizer.step()

            disc_gauss_epoch_loss += disc_gauss_loss.item() * disc_gauss_inputs.size(0)

            ## Encoder - gauss
            enc_gauss_optimizer.zero_grad()
            enc_gauss_outputs = disc_gauss(z_fake)
            enc_gauss_targets = enc_targets.detach()
            enc_gauss_loss = enc_criteria(enc_gauss_outputs, enc_gauss_targets)  # the order cannot be inverted
            enc_gauss_loss.backward()
            enc_gauss_optimizer.step()

            enc_gauss_epoch_loss += enc_gauss_loss.item() * i_size

            ## Discriminator - cat
            _, c_fake = enc(inputs)
            c_real = gen_z_cat(i_size)
            disc_cat_inputs = torch.cat((c_real, c_fake.detach()), dim=0)
            disc_cat_targets = disc_targets.detach()

            disc_cat_optimizer.zero_grad()
            disc_cat_outputs = disc_cat(disc_cat_inputs)
            disc_cat_loss = disc_criteria(disc_cat_outputs, disc_cat_targets)
            disc_cat_loss.backward()
            disc_cat_optimizer.step()

            disc_cat_epoch_loss += disc_cat_loss.item() * disc_cat_inputs.size(0)

            ## Encoder - cat
            enc_cat_optimizer.zero_grad()
            enc_cat_outputs = disc_cat(c_fake)
            enc_cat_targets = enc_targets.detach()
            enc_cat_loss = enc_criteria(enc_cat_outputs, enc_cat_targets)
            enc_cat_loss.backward()
            enc_cat_optimizer.step()

            enc_cat_epoch_loss += enc_cat_loss.item() * i_size

            #########################
            # semi supervised phase #
            #########################
            enc_classifier_optimizer.zero_grad()
            _, predictions = enc(inputs)
            enc_classifier_loss = enc_classifier_criteria(predictions, targets)
            enc_classifier_loss.backward()
            enc_classifier_optimizer.step()

            enc_classifier_epoch_loss = enc_classifier_loss.item() * i_size
            tmp = predictions.detach().cpu().numpy()
            INDEX.add_fake_targets(list(np.argmax(tmp, axis=1)))


            ## record results
            if record % opt.sample_interval == 0:
                # ae_outputs.data = ae_outputs.data.mul(0.5).add(0.5)
                vutils.save_image(ae_outputs.view(-1, 1, opt.imageSize, opt.imageSize),
                                  '{0}/ae_outputs_{1}.png'.format(opt.experiment, record))
                # inputs.data = inputs.data.mul(0.5).add(0.5)
                vutils.save_image(inputs.view(-1, 1, opt.imageSize, opt.imageSize),
                                  '{0}/ae_inputs_{1}.png'.format(opt.experiment, record))
                plot_tsne(z_fake.detach().cpu().numpy(), targets.detach().cpu().numpy().flatten(),
                          "tsne for latent space",
                          "{0}/tsne_z{1}.png".format(opt.experiment, record))
                plot_2D_scatter(z_fake.detach().cpu().numpy(), targets.detach().cpu().numpy().flatten(),
                          "2D for latent space",
                          "{0}/2D_z{1}.png".format(opt.experiment, record))

            record += 1


        ## End of epoch
        ae_epoch_loss /= opt.dataSize
        disc_gauss_epoch_loss /= (opt.dataSize * 2)
        enc_gauss_epoch_loss /= opt.dataSize
        disc_cat_epoch_loss /= (opt.dataSize * 2)
        enc_cat_epoch_loss /= opt.dataSize
        enc_accuracy = INDEX.classification_accuracy()
        # INDEX.plot_latent_distribution("tsne for latent space",
        #                   "{0}/tsne_z.png".format(opt.experiment))

        t.set_postfix(ae=ae_epoch_loss,
                      disc_gauss=disc_gauss_epoch_loss,
                      enc_gauss=enc_gauss_epoch_loss,
                      disc_cat=disc_cat_epoch_loss,
                      enc_cat=enc_cat_epoch_loss,
                      enc_classifier=enc_cat_epoch_loss,
                      enc_accuracy=enc_accuracy)

        writer.add_scalar("ae_loss", ae_epoch_loss, e)
        writer.add_scalar("disc_gauss", disc_gauss_epoch_loss, e)
        writer.add_scalar("enc_gauss", enc_gauss_epoch_loss, e)
        writer.add_scalar("disc_cat", disc_cat_epoch_loss, e)
        writer.add_scalar("enc_cat", enc_cat_epoch_loss, e)
        writer.add_scalar("enc_classifier", enc_cat_epoch_loss, e)
        writer.add_scalar("enc_accuracy", enc_accuracy, e)

# save model parameters
torch.save(enc.state_dict(), '{0}/enc.pth'.format(opt.experiment))
torch.save(dec.state_dict(), '{0}/dec.pth'.format(opt.experiment))
torch.save(disc_cat.state_dict(), "{0}/disc_cat.pth".format(opt.experiment))
torch.save(disc_gauss.state_dict(), "{0}/disc_gauss.pth".format(opt.experiment))

writer.close()

