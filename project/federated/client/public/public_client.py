import torch
import torch.nn as nn
import torch.utils.data
from torchvision.datasets.mnist import MNIST


from federated.client.base_client import BaseClient


class PublicClient(BaseClient):
    """
    Client is responsible for training an epoch and returning its updated
    weights
    """

    def __init__(
        self,
        id: str,
        model: nn.Module,
        device: torch.device,
        data_loader: torch.utils.data.DataLoader[MNIST],
    ):
        _optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
        )
        super().__init__(
            id=id,
            model=model,
            device=device,
            data_loader=data_loader,
            optimizer=_optimizer,
        )

    def train_communication_round(
        self,
        L: int,
        is_verbose: bool,
    ):
        mean_loss, tr_acc = self._train_communication_round(self._data, L, is_verbose)

        return mean_loss, tr_acc, None, None
