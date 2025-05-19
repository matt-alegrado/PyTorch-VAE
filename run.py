import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from multiprocessing import freeze_support

import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torch
# class FIDCallback(pl.callbacks.Callback):
#     def __init__(self, real_images, device):
#         super().__init__()
#         if len(device) > 0:
#             self.fid = FrechetInceptionDistance(feature=64).to('cuda')  # You can change the feature layer
#         else:
#             self.fid = FrechetInceptionDistance(feature=64)
#         self.real_images = real_images  # Pre-computed or loaded real images
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         # pass
#         pl_module.eval()
#         with torch.no_grad():
#             z = torch.randn(16, pl_module.model.latent_dim).to(pl_module.device)
#             generated_images = pl_module.model.decode(z)
#             generated_images_uint8 = (generated_images.clamp(0, 1) * 255).to(torch.uint8)
#
#         self.fid.update(generated_images_uint8, real=False)
#         self.fid.update(self.real_images, real=True)
#         fid_score = self.fid.compute()
#         self.fid.reset()
#         trainer.logger.experiment.add_scalar("FID", fid_score, trainer.current_epoch)

class FIDISCallback(pl.callbacks.Callback):
    def __init__(self, real_images, device):
        super().__init__()

        self.device_str = 'cuda' if len(device) > 0 else 'cpu'
        self.fid = FrechetInceptionDistance(feature=64).to(self.device_str)
        self.inception = InceptionScore(splits=10, normalize=True).to(self.device_str)

        self.real_images = real_images.to(self.device_str)

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            # Generate samples from random latent vectors
            z = torch.randn(128, pl_module.model.latent_dim).to(pl_module.device)
            generated_images = pl_module.model.decode(z)

            # Rescale to [0, 255] and convert to uint8
            generated_images_uint8 = generated_images.to(torch.uint8)

        # Update metrics
        self.fid.update(generated_images_uint8, real=False)
        self.fid.update(self.real_images, real=True)

        self.inception.update(generated_images.float() / 255)  # IS expects float images in [0, 1]

        # Compute metrics
        fid_score = self.fid.compute()
        is_mean, is_std = self.inception.compute()

        # Reset for next epoch
        self.fid.reset()
        self.inception.reset()

        # Log to TensorBoard
        trainer.logger.experiment.add_scalar("FID", fid_score.item(), trainer.current_epoch)
        trainer.logger.experiment.add_scalar("InceptionScore/Mean", is_mean.item(), trainer.current_epoch)
        trainer.logger.experiment.add_scalar("InceptionScore/Std", is_std.item(), trainer.current_epoch)


def main():
    # Force Gloo so NCCL is never touched
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    init_method = "tcp://127.0.0.1:12355"


    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                   name=config['model_params']['name'],)

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                              config['exp_params'])

    data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

    data.setup()
    real_images = next(iter(data.val_dataloader()))[0].to('cuda')
    real_images = (real_images.clamp(0, 1) * 255).to(torch.uint8)
    fidis_callback = FIDISCallback(real_images, config['trainer_params']['gpus'])
    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath =os.path.join(tb_logger.log_dir , "checkpoints"),
                                         monitor= "val_loss",
                                         save_last= True),
                         fidis_callback
                     ],
                     # strategy=DDPPlugin(find_unused_parameters=False),
                     # strategy=DDPStrategy(find_unused_parameters=False),
                     **config['trainer_params'])


    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Load the image
    img = Image.open('logs/VanillaVAE/version_25/Reconstructions/recons_VanillaVAE_Epoch_0.png')

    # Convert to NumPy array
    img_np = np.array(img)

    print("Info for recon ver 25 png")
    print("Image shape:", img_np.shape)
    print("Min pixel value:", img_np.min())
    print("Max pixel value:", img_np.max())
    print("Data type:", img_np.dtype)

    import time

    start_time = time.perf_counter()


    freeze_support()  # needed for Windows spawn
    main()

    # tensorboard --logdir logs
    # ^ this will let me see all the score plots

    import ctypes


    def flash_taskbar():
        FLASHW_ALL = 3
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        ctypes.windll.user32.FlashWindow(hwnd, True)


    flash_taskbar()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")