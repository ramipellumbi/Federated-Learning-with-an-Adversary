from typing import List, Union

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

TClient = Union[
    PublicClient, AdversarialClient, PrivateClient, PrivateAdversarialClient
]


def train_model(*, server: Server, num_rounds: int, L: int, clients: List[TClient]):
    """
    Trains a federated learning model over a specified number of rounds, with a given set of clients.

    This function iterates through a number of training rounds, where in each round, every client updates its model parameters based on the server's model, trains on its local data, and then sends its updated parameters back to the server. The server aggregates these parameters to update its model. The server's model is evaluated periodically and after the final round.

    Parameters:
        server (Server): The federated learning server that coordinates the training process.
        num_rounds (int): The number of rounds to train the model.
        num_clients (int): The number of clients participating in the training process.
        L (int): The number of batches each client should train on its local data per round.
        clients (List[TClient]): A list of clients participating in the training. Each client can be of type PublicClient, AdversarialClient, PrivateClient, or PrivateAdversarialClient.
    """
    num_clients = len(clients)

    # Run 50 epochs on each client
    for round in range(num_rounds):
        # train each client (not done in parallel here but can be done in parallel in practice)
        for i in range(num_clients):
            current_client = clients[i]
            current_client.update_parameters(server.model.state_dict())
            current_client.train_round(L)
            server.add_client_parameters(
                current_client.model.state_dict(),
                current_client.num_samples,
            )
        server.aggregate_parameters()
        if round % 2 == 0 or round == num_rounds - 1:
            server.evaluate_model()


def test_model(
    *,
    model: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader[MNIST],
):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the test images: {100 * correct / total:.2f}%"
        )