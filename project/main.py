import argparse
import os
import sys

import pandas as pd
import torch

from project.federated.client.public_client import PublicClient
from project.federated.client.private_client import PrivateClient
from project.federated.server import Server
from project.data_loaders.mnist.data_loader import DataLoader
from project.models.mnist.mnist_cnn import MnistCNN as Model

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Append the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)


if __name__ == "__main__":
    mps_device = torch.device("mps")

    # get inputs from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=1)
    parser.add_argument("--n_adv", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--b", type=int, default=64)
    parser.add_argument("--p", type=bool, default=False)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args()

    n_clients = args.n_clients
    n_adv = args.n_adv
    num_epochs = args.n_epochs
    batch_size = args.b

    should_use_private = args.p
    if should_use_private:
        if not args.eps or not args.delta:
            raise ValueError(
                "If you want to use private clients, you must provide epsilon and delta"
            )
    target_epsilon = args.eps
    target_delta = args.delta

    dataloader = DataLoader(
        batch_size=batch_size,
        device=mps_device,
        test_split=0.2,
        val_split=0.2,
        n_clients=n_clients,
        num_adversaries=n_adv,
    )

    server = Server(
        model=Model(),
        device=mps_device,
        validation_data=dataloader.val_loader,
        enable_adversary_protection=False,
    )

    clients = [
        PublicClient(
            id=f"{i}",
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
        for i in range(n_clients)
    ]

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
    client_save_name = f"project/results/n_clients_{n_clients}_n_adv_{n_adv}_n_epochs_{num_epochs}_batch_size_{batch_size}_private_{should_use_private}_eps_{target_epsilon}_delta_{target_delta}.csv"
    df.to_csv(client_save_name, index=False)
    server_save_name = f"project/results/n_clients_{n_clients}_n_adv_{n_adv}_n_epochs_{num_epochs}_batch_size_{batch_size}_private_{should_use_private}_eps_{target_epsilon}_delta_{target_delta}_server.csv"
    server.val_performance.to_csv(server_save_name, index=False)
