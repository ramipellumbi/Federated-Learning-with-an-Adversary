from typing import List

import torch
import torch.utils.data
from torchvision.datasets import MNIST

from federated.federated.client import TClient
from federated.federated.server import Server


def train_model(
    *,
    server: Server,
    num_rounds: int,
    num_internal_rounds: int,
    clients: List[TClient],
    is_verbose: bool,
    patience: int = 5,
    min_delta: float = 0.001,
):
    """
    Trains a federated learning model over a specified number of rounds, with a given set of clients.

    This function iterates through a number of training rounds, where in each round, every client updates its model parameters based on the server's model, trains on its local data, and then sends its updated parameters back to the server. The server aggregates these parameters to update its model. The server's model is evaluated periodically and after the final round.

    Parameters:
        server (Server): The federated learning server that coordinates the training process.
        num_rounds (int): The number of rounds to train the model.
        num_clients (int): The number of clients participating in the training process.
        L (int): The number of batches each client should train on its local data per round.
        clients (List[TClient]): A list of clients participating in the training. Each client can be of type PublicClient, AdversarialClient, PrivateClient, or PrivateAdversarialClient.
        is_verbose (bool): A flag indicating whether to print verbose output during training.
        patience (int): The number of rounds to wait for validation loss to improve before early stopping.
        min_delta (float): The minimum change in validation loss to qualify as an improvement.
    """
    num_clients = len(clients)
    best_score = float("inf")
    rounds_without_improvement = 0

    # Run 50 epochs on each client
    for round in range(num_rounds):
        # train each client (not done in parallel here but can be done in parallel in practice)
        for i in range(num_clients):
            current_client = clients[i]
            current_client.update_parameters(server.model.state_dict())
            current_client.train_round(num_internal_rounds, is_verbose)
            server.add_client_parameters(
                current_client.model.state_dict(),
                current_client.num_batches if num_internal_rounds == -1 else num_internal_rounds,
            )

        server.aggregate_parameters(is_verbose)
        if round % 5 == 0 or round == num_rounds - 1:
            print(f"Round {round + 1} of {num_rounds}")
            val_acc = server.evaluate_model()

            if val_acc < best_score - min_delta:
                best_score = val_acc
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if rounds_without_improvement >= patience:
                print(f"Early stopping at round {round + 1}")
                break

            if is_verbose:
                print(f"Best Score: {best_score}, Current Score: {val_acc}")


def test_model(
    *,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader[MNIST],
):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the test images: {100 * correct / total:.2f}%")
