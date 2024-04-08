import time

import tqdm
import wandb
import numpy as np
import cv2
import torch

from losses import mse_loss, mae


class Trainer():
    def __init__(self, model, optimizers, train_loader, val_loader, epochs, sample_num, device, save_model, use_wandb=False):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(list(filter(
            lambda p: p.requires_grad, self.model.parameters()
        )), **optimizers)
        self.model.set_loss(mse_loss)
        self.model.set_new_noise_schedule(device)
        self.sample_num = sample_num
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_model = save_model
        self.use_wandb = use_wandb

    def train_step(self):
        self.model.train()
        losses = []
        for original_gray, gray, color in tqdm(self.train_loader):
            cond_gray = gray.to(self.device)
            gt_color = color.to(self.device)

            self.optimizer.zero_grad()

            loss = self.model(gt_color, cond_gray)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
        return sum(losses)/len(losses)

    def val_step(self, epoch):
        self.model.eval()
        losses, metrics = [], []
        pred_images = []
        gt_images = []

        with torch.no_grad():
            for i, (original_gray, gray, color) in tqdm(enumerate(self.val_loader)):
                cond_gray = gray.to(self.device)
                gt_color = color.to(self.device)
                loss = self.model(gt_color, cond_gray)

                output, visuals = self.model.restoration(
                    cond_gray, sample_num=self.sample_num)
                if i == 0:
                    for i in range(output.shape[0]):
                        # Show predicted image
                        pred_bgr_image = self.show_wandb_image(original_gray[i], output[i].detach().cpu())
                        pred_wandb_image = wandb.Image(pred_bgr_image, caption=f"epoch {epoch}")
                        pred_images.append(pred_wandb_image)

                        # Show grouth truth image
                        gt_bgr_image = self.show_wandb_image(original_gray[i], color[i])
                        gt_wandb_image = wandb.Image(gt_bgr_image, caption=f"epoch {epoch}")
                        gt_images.append(gt_wandb_image)

                mae_score = mae(gt_color, output)
                losses.append(loss.item())
                metrics.append(mae_score.item())
        return sum(losses)/len(losses), sum(metrics)/len(metrics), pred_images, gt_images


    def show_wandb_image(self, img_l, img_ab, is_save=False):
        img_l = img_l.permute(1, 2, 0).numpy()
        img_ab = img_ab.permute(1, 2, 0).numpy()
        img_ab = cv2.resize(img_ab, (img_l.shape[1], img_l.shape[0]), interpolation=cv2.INTER_LINEAR)
        arr_lab = np.concatenate([img_l, img_ab], axis=2)
        arr_lab = (arr_lab + 1.0) * 255 / 2
        arr_lab = np.clip(arr_lab, 0, 255).astype(np.uint8)
        arr_bgr = cv2.cvtColor(arr_lab, cv2.COLOR_LAB2RGB)
        # arr_bgr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
        # arr_bgr = Image.fromarray((arr_bgr * 255).astype(np.uint8))

        return arr_bgr


    def train(self):
        best_mae = 100000
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            train_loss = self.train_step()
            val_loss, val_mae, pred_images, gt_images = self.val_step(epoch)

            if self.use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "pred_images": pred_images,
                    "gt_images": gt_images
                })
            if val_mae < best_mae:
                torch.save(self.model.state_dict(), self.save_model)
            # Print loss, acc end epoch
            print("-" * 59)
            print(
                "| End of epoch {:3d} | Time: {:5.2f}s | Train Loss {:8.3f} "
                "| Valid Loss {:8.3f} | Valid MAE {:8.3f} ".format(
                    epoch+1, time.time() - epoch_start_time, train_loss, val_loss, val_mae
                )
            )
            print("-" * 59)
        self.model.load_state_dict(torch.load(self.save_model))