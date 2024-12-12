from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def expand_t(t, x):
    for _ in range(x.ndim - 1):
        t = t.unsqueeze(-1)
    return t


class FMScheduler(nn.Module):
    def __init__(self, num_train_timesteps=1000, sigma_min=0.001):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min

    def uniform_sample_t(self, batch_size) -> torch.LongTensor:
        ts = (
            np.random.choice(np.arange(self.num_train_timesteps), batch_size)
            / self.num_train_timesteps
        )
        return torch.from_numpy(ts)

    def compute_psi_t(self, x1, t, x):
        """
        Compute the conditional flow psi_t(x | x_1).

        Note that time flows in the opposite direction compared to DDPM/DDIM.
        As t moves from 0 to 1, the probability paths shift from a prior distribution p_0(x)
        to a more complex data distribution p_1(x).

        Input:
            x1 (`torch.Tensor`): Data sample from the data distribution.
            t (`torch.Tensor`): Timestep in [0,1).
            x (`torch.Tensor`): The input to the conditional psi_t(x).
        Output:
            psi_t (`torch.Tensor`): The conditional flow at t.
        """
        t = expand_t(t, x1)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute psi_t(x)

        psi_t = (1 - (1 - self.sigma_min) * t) * x + t * x1
        ######################

        return psi_t

    def step(self, xt, vt, dt):
        """
        The simplest ode solver as the first-order Euler method:
        x_next = xt + dt * vt
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # implement each step of the first-order Euler method.
        x_next = xt + dt * vt
        ######################

        return x_next


class FlowMatching(nn.Module):
    def __init__(self, network: nn.Module, fm_scheduler: FMScheduler, **kwargs):
        super().__init__()
        self.network = network
        self.fm_scheduler = fm_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    def get_loss(self, x1, class_label=None, x0=None):
        """
        The conditional flow matching objective, corresponding Eq. 23 in the FM paper.
        """
        batch_size = x1.shape[0]
        t = self.fm_scheduler.uniform_sample_t(batch_size).to(x1)
        if x0 is None:
            x0 = torch.randn_like(x1)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Implement the CFM objective.
        psi_t = self.fm_scheduler.compute_psi_t(x1, t, x0)
        if class_label is not None:
            model_out = self.network(psi_t, t, class_label=class_label)
        else:
            model_out = self.network(psi_t, t)
        linear_flow = x1 - (1 - self.fm_scheduler.sigma_min) * x0
        loss = F.mse_loss(model_out, linear_flow)
        ######################

        return loss

    def conditional_psi_sample(self, x1, t, x0=None):
        if x0 is None:
            x0 = torch.randn_like(x1)
        return self.fm_scheduler.compute_psi_t(x1, t, x0)

    @torch.no_grad()
    def sample(
        self,
        shape,
        num_inference_timesteps=50,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
        verbose=False,
    ):
        batch_size = shape[0]
        x_T = torch.randn(shape).to(self.device)
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            assert class_label is not None
            assert (
                len(class_label) == batch_size
            ), f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"

        traj = [x_T]

        timesteps = [
            i / num_inference_timesteps for i in range(num_inference_timesteps)
        ]
        timesteps = [torch.tensor([t] * x_T.shape[0]).to(x_T) for t in timesteps]
        pbar = tqdm(timesteps) if verbose else timesteps
        xt = x_T
        for i, t in enumerate(pbar):
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else torch.ones_like(t)

            ######## TODO ########
            # Complete the sampling loop
            if class_label is not None and (guidance_scale != 0.0):
                null_class_label = torch.zeros_like(class_label)
                class_input = torch.cat([null_class_label, class_label], dim=0)
                xt_input = torch.cat([xt, xt], dim=0)
                t_input = torch.cat([t, t], dim=0)
                vt = self.network(xt_input, t_input, class_label=class_input)
                class_vt = vt[batch_size:]
                null_vt = vt[:batch_size]
                vt = (1 - guidance_scale) * null_vt + guidance_scale * class_vt
            else:
                vt = self.network(xt, t)
            dt = (t_next - t).reshape(-1, 1, 1, 1)

            xt = self.fm_scheduler.step(xt, vt, dt)

            ######################

            traj[-1] = traj[-1].cpu()
            traj.append(xt.clone().detach())
        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "fm_scheduler": self.fm_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.fm_scheduler = hparams["fm_scheduler"]

        self.load_state_dict(state_dict)
