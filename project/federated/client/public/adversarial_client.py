from typing import Callable

import torch
import torch.nn as nn
import torch.utils.data

from .public_client import PublicClient


class AdversarialClient(PublicClient):
    def __init__(
        self,
        *,
        id: str,
        model: nn.Module,
        device: torch.device,
        data_loader: torch.utils.data.DataLoader,
        attack: Callable[[nn.Module], nn.Module],
    ):
        super().__init__(id=id,
                         model=model,
                         device=device,
                         data_loader=data_loader)
        self.set_attack(attack)

    def train_communication_round(self, L: int):
        assert (
            self._attack is not None
        ), "Weight attack is not set for adversarial client"

        return super().train_communication_round(L)
