from warnings import warn

import torch
from torch._C import device
from typing_extensions import Annotated
import wandb
from pytorch_lightning import Callback
from PIL import Image, ImageDraw, ImageFont

try:
    import torchvision
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')


class WandbGenerativeModelImageSampler(Callback):
    def __init__(self, num_samples: int = 3, fixed: bool = True, annotate: bool = False):
        """
        Generates images and logs to W&B.
        Your model must implement the forward function for generation

        Requirements::

            # model must have img_dim arg
            model.img_dim = (1, 28, 28)

            # model forward must work for sampling
            z = torch.rand(batch_size, latent_dim)
            img_samples = your_model(z)

        Example::

            from pl_bolts.callbacks import WandbGenerativeModelImageSampler

            # inference with fixed z (noise) and annotate epoch on the image 
            trainer = Trainer(callbacks=[WandbGenerativeModelImageSampler(fixed=True, annotate=True)])
        """
        super().__init__()
        self.num_samples = num_samples
        self.fixed = fixed
        self.annotate = annotate

        self.z = None

    def _compose_images(self, images, trainer, pl_module):
        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)
        grid = torchvision.utils.make_grid(images)
        ndarr = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        im = Image.fromarray(ndarr)
        
        if self.annotate:
            label = f"Epoch: {trainer.current_epoch:003d}"
            d = ImageDraw.Draw(im)
            fnt = ImageFont.load_default()
            x, _ = im.size
            y = 5

            # iteration label
            w, h = fnt.getsize(label)
            d.rectangle((x - w - 4, y, x - 2, y + h), fill="black")
            d.text((x - w - 2, y), label, fnt=fnt, fill=(255, 255, 0))
            
        return im


    def on_epoch_end(self, trainer, pl_module):
        dim = (self.num_samples, pl_module.hparams.latent_dim)

        if self.fixed:
            if trainer.current_epoch == 0:
                self.z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)
            z = self.z
        else:
            z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(z)
            pl_module.train()

        images = self._compose_images(images, trainer, pl_module)
        str_title = f'{pl_module.__class__.__name__}_images'

        trainer.logger.experiment.log({"examples": [wandb.Image(images, caption=str_title)]})