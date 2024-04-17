from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Tuple, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torchvision.datasets.mnist import MNIST


class BaseClient(ABC):
    def __init__(
        self,
        *,
        id: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        data: torch.utils.data.DataLoader[MNIST],
    ):
        self._id = id
        self._device = device
        self._model = model
        self._model.to(self._device)

        self._loss = nn.CrossEntropyLoss()
        self._optimizer = optimizer
        self._data = data

        # each client should track how many epochs they have trained
        self._epochs_trained = 0
        self._num_samples = len(data)

        self._train_history: List[Dict[str, Union[str, Optional[float]]]] = []

    @property
    def train_history(self) -> pd.DataFrame:
        """
        Return dataframe of training history
        """
        df = pd.DataFrame(self._train_history)

        return df

    @property
    def model(self):
        """
        Return the model the client currently has
        """
        return self._model

    @property
    def num_samples(self):
        """
        Return the total number of samples the client has
        """
        return self._num_samples

    def _get_training_accuracy(self):
        """
        Accuracy of the current model on the training data
        """
        self._model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in self._data:
                x, y = x.to(self._device), y.to(self._device)
                yhat = self._model(x)
                correct += (yhat.argmax(1) == y).sum().item()
        return correct / self._num_samples

    @abstractmethod
    def train_one_epoch(
        self,
    ) -> Tuple[float, float, Optional[float], Optional[float]]:
        """
        Train one epoch on the client
        """

    @abstractmethod
    def log_epoch(
        self,
        *,
        loss: float,
        training_accuracy: float,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> None:
        """
        Info logged after epoch completion
        """

    @abstractmethod
    def update_parameters(self, server_state_dict: Dict[str, Any]):
        """Update local model parameters to state of global model.

        Args:
            server_state_dict: the state dictionary of the server model
        """
        self._model.load_state_dict(server_state_dict, strict=True)

    def train_epoch(self):
        """
        Run a federated learning training epoch on the client
        """
        self._model.train()
        loss, tr_acc, epsilon, delta = self.train_one_epoch()
        self._epochs_trained += 1
        self._train_history.append(
            {
                "client": self._id,
                "epoch": self._epochs_trained,
                "loss": loss,
                "training_accuracy": tr_acc,
                "epsilon": epsilon,
                "delta": delta,
            }
        )
        self.log_epoch(
            loss=loss, training_accuracy=tr_acc, epsilon=epsilon, delta=delta
        )
