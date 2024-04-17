import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from project.federated.client.public_client import PublicClient
from project.federated.client.private_client import PrivateClient
from project.data_loaders.mnist.data_loader import DataLoader
from project.federated.server import Server
from project.models.mnist.mnist_cnn import MnistCNN as Model

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Append the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)


if __name__ == "__main__":
    mps_device = torch.device("mps")

    n_clients = 1
    n_adv = 0
    num_epochs = 2
    batch_size = 64

    dataloader = DataLoader(
        batch_size=batch_size,
        device=mps_device,
        val_split=0.2,
        n_clients=n_clients,
        num_adversaries=n_adv,
    )
    aggregator = Server(
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
        for i in range(n_clients)
    ]

    # Run 50 epochs on each client
    for epoch in range(num_epochs):
        # train each client
        for i in range(n_clients):
            current_client = clients[i]
            current_client.update_parameters(aggregator.model.state_dict())
            current_client.train_epoch()
            aggregator.add_client_parameters(
                current_client.model.state_dict(),
                current_client.num_samples,
            )
        aggregator.aggregate_parameters()
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            aggregator.evaluate_model()

    # evaluate the model on the test set

    with torch.no_grad():
        aggregator.model.eval()
        correct = 0
        total = 0
        for images, labels in dataloader.test_loader:
            images, labels = images.to(mps_device), labels.to(mps_device)
            outputs = aggregator.model(images)
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
    df.to_csv("non_private_train_history.csv", index=False)
    aggregator.val_performance.to_csv("non_private_val_performance.csv", index=False)
