import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from pl_bolts.models.gans.basic.components import Generator, Discriminator


class GAN(pl.LightningModule):

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        latent_dim: int = 32,
        learning_rate: float = 0.0002,
        **kwargs
    ):
        """
        Vanilla GAN implementation.

        Example::

            from pl_bolts.models.gan import GAN

            m = GAN()
            Trainer(gpus=2).fit(m)

        Args:

            datamodule: the datamodule (train, val, test splits)
            latent_dim: emb dim for encoder
            batch_size: the batch size
            learning_rate: the learning rate
            data_dir: where to store data
            num_workers: data workers

        """
        super().__init__()

        # makes self.hparams under the hood and saves to ckpt
        self.save_hyperparameters()
        self.img_dim = (input_channels, input_height, input_width)

        # networks
        self.generator = self.init_generator(self.img_dim)
        self.discriminator = self.init_discriminator(self.img_dim)

    def init_generator(self, img_dim):
        generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=img_dim)
        return generator

    def init_discriminator(self, img_dim):
        discriminator = Discriminator(img_shape=img_dim)
        return discriminator

    def forward(self, z):
        """
        Generates an image given input noise z

        Example::

            z = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(z)
        """
        return self.generator(z)

    def generator_loss(self, x):
        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_dim, device=self.device)
        y = torch.ones(x.size(0), 1, device=self.device)

        # generate images
        generated_imgs = self(z)

        D_output = self.discriminator(generated_imgs)

        # ground truth result (ie: all real)
        g_loss = F.binary_cross_entropy(D_output, y)

        return g_loss

    def discriminator_loss(self, x):
        # train discriminator on real
        b = x.size(0)
        x_real = x.view(b, -1)
        y_real = torch.ones(b, 1, device=self.device)

        # calculate real score
        D_output = self.discriminator(x_real)
        D_real_loss = F.binary_cross_entropy(D_output, y_real)

        # train discriminator on fake
        z = torch.randn(b, self.hparams.latent_dim, device=self.device)
        x_fake = self(z)
        y_fake = torch.zeros(b, 1, device=self.device)

        # calculate fake score
        D_output = self.discriminator(x_fake)
        D_fake_loss = F.binary_cross_entropy(D_output, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss

        return D_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch

        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_step(x)

        return result

    def generator_step(self, x):
        g_loss = self.generator_loss(x)

        # log to prog bar on each step AND for the full epoch
        # use the generator loss for checkpointing
        result = pl.TrainResult(minimize=g_loss, checkpoint_on=g_loss)
        result.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return result

    def discriminator_step(self, x):
        # Measure discriminator's ability to classify real from generated samples
        d_loss = self.discriminator_loss(x)

        # log to prog bar on each step AND for the full epoch
        result = pl.TrainResult(minimize=d_loss)
        result.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        return result

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

