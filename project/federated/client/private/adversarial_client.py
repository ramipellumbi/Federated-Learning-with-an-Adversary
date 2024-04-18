from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch
import torch.nn as nn
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
        data: torch.utils.data.DataLoader,
        max_grad_norm: float,
        num_epochs: int,
        target_epsilon: float,
        target_delta: float,
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

    def train_one_epoch(
        self,
        epsilon: float = 0.3,
    ):
        losses = []
        with BatchMemoryManager(
            data_loader=self._data,
            max_physical_batch_size=64,
            optimizer=self._optimizer,
        ) as memory_safe_loader
            for x, y in tqdm(
                self._data,
                desc=f"{self._id} | Epoch {self._epochs_trained}",
            ):
                x, y = x.to(self._device), y.to(self._device)
                yhat = self._model(x)
                self._optimizer.zero_grad()
                loss = self._loss(yhat, y)
                loss.backward()

                # Adversarially perturbing the gradients based on FGSM
                with torch.no_grad():
                    for param in self._model.parameters():
                        if param.grad is not None:
                            # Instead of altering input, the gradient is modified
                            param.grad += epsilon * -1 * torch.sign(param.grad)

                losses.append(loss.item())
                self._optimizer.step()

            losses = torch.Tensor(losses)
            training_accuracy = self._get_training_accuracy()

            epsilon = self._privacy_engine.accountant.get_epsilon(delta=self._td)

        return losses.mean().item(), training_accuracy, epsilon, self._td
