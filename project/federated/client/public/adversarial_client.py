import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

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
        self._noise_multiplier = noise_multiplier

    def train_one_epoch(
        self,
    ):
        losses = []
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
                        param.grad += (
                            self._noise_multiplier * -1 * torch.sign(param.grad)
                        )

            losses.append(loss.item())
            self._optimizer.step()

        losses = torch.Tensor(losses)
        training_accuracy = self._get_training_accuracy()

        return losses.mean().item(), training_accuracy, None, None
