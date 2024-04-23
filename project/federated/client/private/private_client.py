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

        self.set_model(_model)
        self.set_optimizer(_optimizer)
        self.set_data_loader(_data_loader)

    def train_communication_round(self, L: int):
        assert isinstance(
            self._optimizer, DPOptimizer
        ), "Optimizer must be DPOptimizer for DP training"

        with BatchMemoryManager(
            data_loader=self._data,
            max_physical_batch_size=64,
            optimizer=self._optimizer,
        ) as memory_safe_loader:
            mean_loss, tr_acc = self._train_communication_round(memory_safe_loader, L)
            epsilon = self._privacy_engine.accountant.get_epsilon(delta=self._td)

        return mean_loss, tr_acc, epsilon, self._td

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
        # assert that all keys start with "_module."
        assert all(k.startswith("_module.") for k in server_state_dict.keys())

        super().update_parameters(server_state_dict)
