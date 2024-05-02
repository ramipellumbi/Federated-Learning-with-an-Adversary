from typing import Callable, Optional

from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch
import torch.nn as nn
from torch.nn.modules import Module
import torch.utils.data
from tqdm import tqdm

from .private_client import PrivateClient


class AdversarialClient(PrivateClient):
    def __init__(
        self,
        *,
        id: str,
        model: nn.Module,
        device: torch.device,
        data_loader: torch.utils.data.DataLoader,
        max_grad_norm: float,
        num_epochs: int,
        target_epsilon: float,
        target_delta: float,
        attack: Callable[[nn.Module], nn.Module],
    ):
        super().__init__(
            id=id,
            model=model,
            device=device,
            data_loader=data_loader,
            max_grad_norm=max_grad_norm,
            num_epochs=num_epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
        )
        self.set_attack(attack)

    def train_communication_round(self, L: int):
        assert (
            self._attack is not None
        ), "Weight attack is not set for adversarial client"

        return super().train_communication_round(L)
