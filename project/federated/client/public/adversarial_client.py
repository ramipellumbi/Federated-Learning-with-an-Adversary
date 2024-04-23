from typing import Optional

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from project.attacks.gradient_attack import gradient_attack
from .public_client import PublicClient


class AdversarialClient(PublicClient):
    def __init__(
        self,
        *,
        id: str,
        model: nn.Module,
        device: torch.device,
        data: torch.utils.data.DataLoader,
        noise_multiplier: float,
    ):
        super().__init__(id=id, model=model, device=device, data=data)
        self.set_noise_multiplier(noise_multiplier)
        self.set_weight_attack(gradient_attack)

    def train_communication_round(self, L: int):
        assert (
            self._weight_attack is not None
        ), "Weight attack is not set for adversarial client"
        assert (
            self._noise_multiplier is not None
        ), "Noise multiplier is not set for adversarial client"

        return super().train_communication_round(L)
