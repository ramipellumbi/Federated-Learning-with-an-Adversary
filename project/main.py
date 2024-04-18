import argparse
import os
import sys

import pandas as pd
import torch

from project.federated.client.private.private_client import PrivateClient
from project.federated.client.private.adversarial_client import (
    AdversarialClient as PrivateAdversarialClient,
)
from project.federated.client.public.public_client import PublicClient
from project.federated.client.public.adversarial_client import AdversarialClient
from project.federated.server import Server
from project.data_loaders.mnist.data_loader import DataLoader
from project.models.mnist.mnist_cnn import MnistCNN as Model

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Append the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def validate_command_line_arguments(args):
    # validate inputs
    assert args.n_clients > 0, "Number of clients must be greater than 0"
    assert args.n_adv > 0, "Number of adversaries must be greater than 0"
    assert (
        args.n_adv < args.n_clients
    ), "Number of adversaries must be less than number of clients"
    assert args.noise_multiplier > 0, "Noise multiplier must be greater than 0"
    assert args.n_epochs > 0, "Number of epochs must be greater than 0"
    assert isinstance(
        args.use_differential_privacy, bool
    ), "Use differential privacy must be a boolean"
    assert args.batch_size > 0, "Batch size must be greater than 0"
    assert isinstance(
        args.enable_adv_protection, bool
    ), "Enable adv protection must be a boolean"
    assert args.eps is None or args.eps > 0, "Epsilon must be greater than 0"
    assert args.delta is None or args.delta > 0, "Delta must be greater than 0"

    should_use_private = args.use_differential_privacy
    if should_use_private:
        if not args.eps or not args.delta:
            raise ValueError(
                "If you want to use private clients, you must provide epsilon and delta"
            )


if __name__ == "__main__":
    mps_device = torch.device("mps")

    # get inputs from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--n_adv", type=int, default=2)
    parser.add_argument("--noise_multiplier", type=float, default=0.3)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--enable_adv_protection", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_differential_privacy", type=bool, default=False)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args()

    validate_command_line_arguments(args)

    n_clients = args.n_clients
    n_adv = args.n_adv
    num_epochs = args.n_epochs
    batch_size = args.batch_size
    enable_adv_protection = args.enable_adv_protection
    noise_multiplier = args.noise_multiplier
    target_epsilon = args.eps
    target_delta = args.delta
    should_use_private = args.use_differential_privacy

    dataloader = DataLoader(
        batch_size=batch_size,
        device=mps_device,
        test_split=0.2,
        val_split=0.2,
        n_clients=n_clients,
    )

    server = Server(
        model=Model(),
        device=mps_device,
        validation_data=dataloader.val_loader,
        enable_adversary_protection=enable_adv_protection,
    )

    # create adversarial clients for the first n_adv clients and public clients for the rest
    adv_clients = [
        AdversarialClient(
            id=f"Adversarial Client {i}",
            model=Model(),
            device=mps_device,
            data=dataloader.train_loaders[i],
            noise_multiplier=noise_multiplier,
        )
        if not should_use_private
        else PrivateAdversarialClient(
            id=f"Adversarial Client {i}",
            model=Model(dropout=False),
            device=mps_device,
            data=dataloader.train_loaders[i],
            max_grad_norm=1.0,
            num_epochs=num_epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            noise_multiplier=noise_multiplier,
        )
        for i in range(n_adv)
    ]

    non_adv_clients = [
        (
            PublicClient(
                id=f"Client {i}",
                model=Model(),
                device=mps_device,
                data=dataloader.train_loaders[i],
            )
            if not should_use_private
            else PrivateClient(
                id=f"{i}",
                model=Model(dropout=False),
                device=mps_device,
                data=dataloader.train_loaders[i],
                max_grad_norm=1.0,
                num_epochs=num_epochs,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
            )
        )
        for i in range(n_adv, n_clients)
    ]

    clients = adv_clients + non_adv_clients

    # Run 50 epochs on each client
    for epoch in range(num_epochs):
        # train each client
        for i in range(n_clients):
            current_client = clients[i]
            current_client.update_parameters(server.model.state_dict())
            current_client.train_epoch()
            server.add_client_parameters(
                current_client.model.state_dict(),
                current_client.num_samples,
            )
        server.aggregate_parameters()
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            server.evaluate_model()

    # evaluate the model on the test set

    with torch.no_grad():
        server.model.eval()
        correct = 0
        total = 0
        for images, labels in dataloader.test_loader:
            images, labels = images.to(mps_device), labels.to(mps_device)
            outputs = server.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the test images: {100 * correct / total:.2f}%"
        )

    # save data to csv
    all_dfs = []
    for i in range(n_clients):
        all_dfs.append(clients[i].train_history)

    # combine all dataframes to one dataframe
    df = pd.concat(all_dfs)
    client_save_name = f"project/results/n_clients_{n_clients}_n_adv_{n_adv}_enable_adv_{enable_adv_protection}_n_epochs_{num_epochs}_batch_size_{batch_size}_private_{should_use_private}_eps_{target_epsilon}_delta_{target_delta}.csv"
    df.to_csv(client_save_name, index=False)
    server_save_name = f"project/results/n_clients_{n_clients}_n_adv_{n_adv}_enable_adv_{enable_adv_protection}_n_epochs_{num_epochs}_batch_size_{batch_size}_private_{should_use_private}_eps_{target_epsilon}_delta_{target_delta}_server.csv"
    server.val_performance.to_csv(server_save_name, index=False)
