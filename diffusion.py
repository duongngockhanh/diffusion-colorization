from functools import partial

import numpy as np
import torch
import torch.nn as nn

from model import UNet
from config import *

def make_beta_schedule(schedule, n_timestep, linear_start=1e-5, linear_end=1e-2):
    if schedule == 'linear':
        betas = np.linspace(
            linear_start, linear_end, n_timestep, dtype=np.float64
        )
    else:
        raise NotImplementedError(schedule)
    return betas

def get_index_from_list(vals, t, x_shape=(1,1,1,1)):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size, *_ = t.shape
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)


class ColorDiffusion(nn.Module):
    def __init__(self, unet_config, beta_schedule, **kwargs):
        super(ColorDiffusion, self).__init__(**kwargs)
        self.denoise_fn = UNet(**unet_config)
        self.beta_schedule = beta_schedule

    def set_new_noise_schedule(self, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule)
        alphas = 1. - betas
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0) # alphas_cumprod
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            get_index_from_list(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            get_index_from_list(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        posterior_mean = (
            get_index_from_list(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            get_index_from_list(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = get_index_from_list(
            self.posterior_log_variance_clipped, t, y_t.shape
        )
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = get_index_from_list(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = noise if noise is not None else torch.randn_like(y_0)
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, sample_num=8):
        b, _, h, w = y_cond.shape

        sample_inter = (self.num_timesteps//sample_num)

        y_t = y_t if y_t is not None else torch.randn((b, 2, h, w))
        y_t = y_t.to(y_cond.device)
        ret_arr = y_t
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = get_index_from_list(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = get_index_from_list(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = noise if noise is not None else torch.randn_like(y_0)
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
        loss = self.loss_fn(noise, noise_hat)
        return loss