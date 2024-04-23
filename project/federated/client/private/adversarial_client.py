from typing import Optional

from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from project.attacks.gradient_attack import gradient_attack
from .private_client import PrivateClient


class AdversarialClient(PrivateClient):
    def __init__(
        self,
        *,
        id: str,
        model: nn.Module,
        device: torch.device,
        data: torch.utils.data.DataLoader,
        max_grad_norm: float,
        num_epochs: int,
        target_epsilon: float,
        target_delta: float,
        noise_multiplier: float,
    ):
        super().__init__(
            id=id,
            model=model,
            device=device,
            data=data,
            max_grad_norm=max_grad_norm,
            num_epochs=num_epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
        )
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
