from abc import ABC, abstractmethod
from itertools import cycle, islice
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

from opacus.optimizers import DPOptimizer
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm


class BaseClient(ABC):
    def __init__(
        self,
        *,
        id: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        data_loader: torch.utils.data.DataLoader[MNIST],
    ):
        self._id = id
        self._device = device
        self._model = model
        self._model.to(self._device)

        self._loss = nn.CrossEntropyLoss()
        self._optimizer = optimizer
        self._data = data_loader

        # each client should track how many epochs they have trained
        self._epochs_trained = 0
        self._num_samples = len(data_loader)

        self._train_history: List[Dict[str, Union[str, Optional[float]]]] = []

        self._current_index = 0
        self._dataset_completed = False

        self._attack: Optional[Callable[[nn.Module], nn.Module]] = None

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
        # if there is an attack, return the model with attacked weights
        if self._attack is not None:
            return self._attack(self._model)

        return self._model

    @property
    def num_batches(self):
        """
        Return the total number of batches the client has
        """
        return self._num_samples

    def set_attack(self, attack: Callable[[nn.Module], nn.Module]):
        self._attack = attack

    def set_model(self, model: nn.Module):
        """
        Set the model of the client
        """
        self._model = model
        self._model.to(self._device)

    def set_optimizer(self, optimizer: Union[torch.optim.Optimizer, DPOptimizer]):
        """
        Set the optimizer of the client
        """
        self._optimizer = optimizer

    def set_data_loader(self, data: torch.utils.data.DataLoader[MNIST]):
        """
        Set the data of the client
        """
        self._data = data
        self._num_samples = len(data)

    def _get_training_accuracy(self):
        """
        Accuracy of the current model on the training data
        """
        self._model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in tqdm(self._data, desc=f"{self._id} | Train Accuracy"):
                x, y = x.to(self._device), y.to(self._device)
                yhat = self._model(x)
                correct += (yhat.argmax(1) == y).sum().item()
        return correct / self._num_samples

    def _train_communication_round(
        self, data_loader: torch.utils.data.DataLoader[MNIST], L: int
    ):
        if (
            self._dataset_completed
        ):  # Reset if dataset was fully iterated in previous training
            self._current_index = 0
            self._dataset_completed = False

        self._model.train()
        L = L if L != -1 else len(self._data)
        losses = []

        cyclic_data_loader = cycle(data_loader)
        sliced_data_loader = islice(
            cyclic_data_loader, self._current_index, self._current_index + L
        )

        for x, y in tqdm(sliced_data_loader, desc=f"{self._id} | Training"):
            x, y = x.to(self._device), y.to(self._device)
            yhat = self._model(x)
            self._optimizer.zero_grad()
            loss = self._loss(yhat, y)
            loss.backward()

            losses.append(loss.item())
            self._optimizer.step()

        self._current_index = (self._current_index + L) % self._num_samples
        losses = torch.Tensor(losses)
        training_accuracy = self._get_training_accuracy()

        return losses.mean().item(), training_accuracy

    @abstractmethod
    def train_communication_round(
        self,
        L: int,
    ) -> Tuple[float, float, Optional[float], Optional[float]]:
        """
        Train one epoch on the client

        Args:
            L: number of local steps to take
        """

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
        print_str = f"{self._id} | Epoch {self._epochs_trained} | Loss: {loss:.4f} | Tr. Acc: {training_accuracy:.4f}"
        if epsilon is not None:
            print_str += f" | ε: {epsilon:.4f}"

        if delta is not None:
            print_str += f" | δ: {delta:.4f}"

        print(print_str)

    def update_parameters(self, server_state_dict: Dict[str, Any]):
        """Update local model parameters to state of global model.

        Args:
            server_state_dict: the state dictionary of the server model
        """
        self._model.load_state_dict(server_state_dict, strict=True)

    def train_round(self, L: int):
        """
        Run a federated learning training epoch on the client
        """
        loss, tr_acc, epsilon, delta = self.train_communication_round(L)

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
        self._epochs_trained += 1
