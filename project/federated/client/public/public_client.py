from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.utils.data
from typing import List, Optional
from tqdm import tqdm
from torchvision.datasets.mnist import MNIST


from project.federated.client.base_client import BaseClient


class PublicClient(BaseClient):
    """
    Client is responsible for training an epoch and returning its updated weights
    """

    def __init__(
        self,
        *,
        id: str,
        model: nn.Module,
        device: torch.device,
        data: torch.utils.data.DataLoader[MNIST],
    ):
        _optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
        )
        super().__init__(
            id=id, model=model, device=device, data=data, optimizer=_optimizer
        )

    def train_one_epoch(
        self,
    ):
        losses = []
        self._model.train()
        for x, y in tqdm(
            self._data, desc=f"Client {self._id} | Epoch {self._epochs_trained}"
        ):
            x, y = x.to(self._device), y.to(self._device)
            yhat = self._model(x)
            self._optimizer.zero_grad()
            loss = self._loss(yhat, y)
            loss.backward()
            losses.append(loss.item())
            self._optimizer.step()

        losses = torch.Tensor(losses)
        training_accuracy = self._get_training_accuracy()

        return losses.mean().item(), training_accuracy, None, None

    def log_epoch(
        self,
        *,
        loss: float,
        training_accuracy,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> None:
        print(
            f"Client {self._id} | Epoch {self._epochs_trained} | Loss: {loss:.4f} | Tr. Acc: {training_accuracy:.4f}"
        )

    def update_parameters(self, server_state_dict: Dict[str, Any]):
        super().update_parameters(server_state_dict)
