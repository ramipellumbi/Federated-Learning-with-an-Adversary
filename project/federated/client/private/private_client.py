from typing import Any, Dict, Optional

from opacus import GradSampleModule, PrivacyEngine
from opacus.data_loader import DataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
import torch
import torch.nn as nn
import torch.utils.data
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

from project.federated.client.base_client import BaseClient


class PrivateClient(BaseClient):
    """
    PrivateClient inherits base client and modifies the model,
    optimizer, and dataloader to train with differential privacy
    via Opacus.
    """

    def __init__(
        self,
        *,
        id: str,
        model: nn.Module,
        device: torch.device,
        data: torch.utils.data.DataLoader[MNIST],
        max_grad_norm: float,
        num_epochs: int,
        target_epsilon: float,
        target_delta: float,
    ):
        # # initialize the client
        _private_optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=0.001,  # larger LR for the DP model
            betas=(0.9, 0.999),
        )
        super().__init__(
            id=id,
            model=model,
            device=device,
            data=data,
            optimizer=_private_optimizer,
        )

        self._td = target_delta
        self._max_grad_norm = max_grad_norm

        # csprng does not work with this version of PyTorch & Python.
        # all setting False does is potentially use non secure parallel RNG.
        # Fine for testing.
        self._privacy_engine = PrivacyEngine(secure_mode=False)

        if not ModuleValidator.is_valid(self._model):
            module = ModuleValidator.fix(self._model)
        else:
            module = self._model

        (
            _model,
            _optimizer,
            _data_loader,
        ) = self._privacy_engine.make_private_with_epsilon(
            module=module,
            optimizer=self._optimizer,
            data_loader=self._data,
            epochs=num_epochs,
            target_epsilon=target_epsilon,
            target_delta=self._td,
            max_grad_norm=self._max_grad_norm,
        )

        self._model: GradSampleModule = _model
        self._optimizer: DPOptimizer = _optimizer
        self._data: DataLoader[MNIST] = _data_loader

    # override train one epoch function
    def train_one_epoch(self):
        """Perform one epoch of training on model

        Args:
            train_loader: training set as a torch.utils.data.DataLoader
            device: specify which device to train on
            epoch: epoch being trained
            client: name of client
        """
        losses = []

        with BatchMemoryManager(
            data_loader=self._data,
            max_physical_batch_size=64,
            optimizer=self._optimizer,
        ) as memory_safe_loader:
            for x, y in tqdm(
                memory_safe_loader,
                desc=f"{self._id} | Epoch {self._epochs_trained}",
            ):
                x, y = x.to(self._device), y.to(self._device)
                self._optimizer.zero_grad()
                yhat = self._model(x)
                loss = self._loss(yhat, y)
                loss.backward()
                self._optimizer.step()
                losses.append(loss.item())

            losses = torch.Tensor(losses)
            training_accuracy = self._get_training_accuracy()
            epsilon = self._privacy_engine.accountant.get_epsilon(delta=self._td)

        return losses.mean().item(), training_accuracy, epsilon, self._td

    def log_epoch(
        self,
        *,
        loss: float,
        training_accuracy: float,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ):
        print(
            f"{self._id} | Epoch {self._epochs_trained} | Loss: {loss:.4f} | Tr. Acc: {training_accuracy:.4f} | ε = {epsilon:.4f} | δ = {self._td}"
        )

    def update_parameters(self, server_state_dict: Dict[str, Any]):
        # prepend the model with the privacy engine weight names by adding the prefix "_module to all the keys"
        server_state_dict = {f"_module.{k}": v for k, v in server_state_dict.items()}
        super().update_parameters(server_state_dict)
