import os
import sys
from typing import List

import pandas as pd
import torch
import torch.utils.data
from torchvision.datasets import MNIST

from project.federated.client import (
    PublicClient,
    AdversarialClient,
    PrivateClient,
    PrivateAdversarialClient,
)
from project.federated.server import Server
from project.data_loaders.mnist.data_loader import DataLoader
from project.models.mnist.mnist_cnn import MnistCNN as Model
from project.setup import get_command_line_args
from project.training import train_model, test_model, TClient
from project.utilities import save_results

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Append the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)


if __name__ == "__main__":
    args = get_command_line_args()

    n_clients: int = args.n_clients
    n_adv: int = args.n_adv

    num_rounds: int = args.n_rounds
    L: int = args.L
    batch_size: int = args.batch_size
    use_iid_data: bool = args.should_use_iid_training_data

    enable_adv_protection: bool = args.should_enable_adv_protection
    noise_multiplier: float = args.noise_multiplier

    should_use_private: bool = args.should_use_private_clients
    target_epsilon: float = args.target_epsilon
    target_delta: float = args.target_delta

    mps_device = torch.device("mps")
    dataloader = DataLoader(
        batch_size=batch_size,
        device=mps_device,
        test_split=0.2,
        val_split=0.2,
        n_clients=n_clients,
        use_iid=use_iid_data,
    )
    print(enable_adv_protection)
    server = Server(
        model=Model(),
        device=mps_device,
        validation_data=dataloader.val_loader,
        enable_adversary_protection=enable_adv_protection,
    )

    adv_clients: List[TClient] = [
        AdversarialClient(
            id=f"Adversarial Client {i}",
            model=Model(),
            device=mps_device,
            data=dataloader.train_loaders[i],
            noise_multiplier=noise_multiplier,
        )
        if not should_use_private
        else PrivateAdversarialClient(
            id=f"Private Adversarial Client {i}",
            model=Model(),
            device=mps_device,
            data=dataloader.train_loaders[i],
            noise_multiplier=noise_multiplier,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            num_epochs=num_rounds if L == -1 else num_rounds // L,
            max_grad_norm=100.0,
        )
        for i in range(n_adv)
    ]

    non_adv_clients: List[TClient] = [
        (
            PublicClient(
                id=f"Client {i}",
                model=Model(),
                device=mps_device,
                data=dataloader.train_loaders[i],
            )
        )
        if not should_use_private
        else (
            PrivateClient(
                id=f"Private Client {i}",
                model=Model(),
                device=mps_device,
                data=dataloader.train_loaders[i],
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                num_epochs=num_rounds if L == -1 else num_rounds // L,
                max_grad_norm=100.0,
            )
        )
        for i in range(n_adv, n_clients)
    ]

    clients = adv_clients + non_adv_clients
    assert len(clients) == n_clients

    train_model(
        server=server,
        num_rounds=num_rounds,
        clients=clients,
        L=L,
    )

    test_model(
        model=server.model,
        device=mps_device,
        test_loader=dataloader.test_loader,
    )

    save_results(
        server=server,
        clients=clients,
        config=args,
    )
