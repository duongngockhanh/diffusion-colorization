import gc

import wandb

from diffusion import ColorDiffusion
from trainer import Trainer
from config import *
from dataloader import train_loader, val_loader


# Declare models
colordiff_model = ColorDiffusion(unet_config, beta_schedule)
trainer = Trainer(
    colordiff_model, optimizers,
    train_loader, val_loader,
    epochs, sample_num,
    device, save_model,
    use_wandb
)

# Login wandb
if use_wandb:
    wandb_api="387da1f220b55f23dec29347d30650c011d7ecee"
    wandb.login(key=wandb_api)

# Initialize a session
if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-diff-color",

        # track hyperparameters and run metadata
        config={
        "learning_rate": optimizers["lr"],
        "weight_decay": optimizers["weight_decay"],
        "architecture": "UNet",
        "dataset": "CelebA",
        "epochs": epochs,
        "save_model": save_model,
        "sample_num": sample_num
        }
    )

# Run session
trainer.train()

# Finish session
if use_wandb:
    wandb.finish()

# Clean GPU memory
gc.collect()
torch.cuda.empty_cache()