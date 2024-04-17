from typing import Any, Dict, List, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torchvision.datasets import MNIST


class Server:
    """
    The server is responsible for aggregating the weights of all clients
    and updating accordingly
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        validation_data: torch.utils.data.DataLoader[MNIST],
        enable_adversary_protection: bool = False,
    ):
        # the server never trains, it only aggregates weights into the model
        self._model = model
        self._device = device
        self._model.to(device)

        self._validation_data = validation_data
        self._enable_adversary_protection = enable_adversary_protection

        # client_parameters[i] is the parameters of model trained by client i
        self._client_weights: List[Dict[str, Any]] = []
        # _client_samples_processed[i] is the number of samples client i used for training
        self._client_samples: List[int] = []
        # trust scores for each client
        self._client_trust: Dict[int, float] = {}
        # used to store the weights used in an aggregation - updated each aggregation
        self._client_score: List[float] = []

        # internal validation accuracy for each aggregation round
        self._val_performance: List[Dict[str, Union[str, float]]] = []

    def _calculate_median_weights(self):
        """Calculate the median of weights across clients."""
        num_params = len(self._client_weights[0])
        param_keys = list(self._client_weights[0].keys())
        median_weights = {}

        # For each parameter, stack across clients and calculate median
        for key in param_keys:
            stacked_params = torch.stack(
                [client[key].to("cpu") for client in self._client_weights]
            )
            median, _ = torch.median(stacked_params, dim=0)
            median_weights[key] = median.to(self._device)

        return median_weights

    def _update_trust_scores(self, new_scores: List[float]):
        """
        Update the trust scores for each client based on the new scores.
        """
        for i, score in enumerate(new_scores):
            # use 90% old score and 10% new score
            if i in self._client_trust:
                old_score = self._client_trust[i]
                new_score = 0.9 * old_score + 0.1 * score
                self._client_trust[i] = new_score
            # use new score if no old score
            else:
                self._client_trust[i] = score

        # normalize scores to sum to 1
        total_trust = sum(self._client_trust.values())
        for client_id in self._client_trust:
            self._client_trust[client_id] /= total_trust

    def _compute_trust_scores(self):
        """
        Compute the trust scores for each client based on the deviation of their weights
        from the median.
        """
        median_weights = self._calculate_median_weights()
        anomaly_scores: torch.Tensor = torch.Tensor([]).to(self._device)

        for client_weights in self._client_weights:
            total_deviation = torch.Tensor([0.0]).to(self._device)
            for key, param in client_weights.items():
                median_param = median_weights[key]
                deviation = torch.norm(param - median_param)
                total_deviation += deviation

            anomaly_scores = torch.cat((anomaly_scores, total_deviation))

        max_anomaly = torch.max(anomaly_scores)
        new_scores = (1 - anomaly_scores / max_anomaly).squeeze().tolist()
        self._update_trust_scores(new_scores)

    @property
    def val_performance(self) -> pd.DataFrame:
        """
        Return the validation performance of the server model.
        """
        return pd.DataFrame(self._val_performance)

    @property
    def model(self):
        """
        Grab the current model from the server.
        """
        return self._model

    def evaluate_model(self):
        """
        Evaluate the model on the validation set and log the checkpointing.
        """
        # assess performance on validation set
        self._model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self._validation_data:
                images, labels = images.to(self._device), labels.to(self._device)
                outputs = self._model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the server validation images: {100 * correct / total:.2f}%"
        )

        # add a row for each client to the performance log
        for i in range(len(self._client_score)):
            self._val_performance.append(
                {
                    "accuracy": 100 * correct / total,
                    "client": i,
                    "client_trust_score": self._client_trust[i],
                    "client_score": self._client_score[i],
                }
            )

    def add_client_parameters(
        self,
        client_state_dict: Dict[str, Any],
        num_samples: int,
    ):
        """
        Add the parameters of a client to the server and the number of samples it used for training.
        """
        client_id = len(self._client_weights)
        self._client_weights.append(client_state_dict)
        self._client_samples.append(num_samples)

        # Initialize trust score uniformly among all known clients
        if client_id not in self._client_trust:
            # Reset all to equal trust
            for i in range(len(self._client_weights)):
                self._client_trust[i] = 1.0 / len(self._client_weights)

    def aggregate_parameters(
        self,
    ):
        """
        Aggregate the parameters of all clients and update the model with the aggregated weights.

        If adversary_protection is True, the server will compute trust scores for each client
        and reweight their contributions to the aggregation.
        """
        self._model.train()

        # if we are using adversary protection, compute trust scores to reweight client contributions
        if self._enable_adversary_protection:
            self._compute_trust_scores()

        total_samples = sum(self._client_samples)
        aggregated_weights = {
            name: torch.zeros_like(param, device=self._device)
            for name, param in self._model.named_parameters()
        }

        # this is the ratio of data client i has compared to all clients -- sums to 1
        client_ratios = [
            self._client_samples[i] / total_samples
            for i in range(len(self._client_samples))
        ]

        # this is the ratio of data client i has compared to all clients weighted by trust -- does NOT sum to 1 necessarily
        new_contribution = [
            self._client_trust[i] * client_ratios[i]
            for i in range(len(self._client_trust))
        ]

        # normalize contributions to sum to 1
        total_new_contribution = sum(new_contribution)
        client_contributions = [
            contribution / total_new_contribution for contribution in new_contribution
        ]
        self._client_score = client_contributions.copy()
        print([c for c in client_contributions])

        with torch.no_grad():
            for client_weight, weight_fraction in zip(
                self._client_weights, client_contributions
            ):
                for name, param in client_weight.items():
                    # dp waits are prepended with _module. -- strip this for global model
                    name = name.replace("_module.", "")
                    aggregated_weights[name] += param.clone().detach() * weight_fraction

        self._model.load_state_dict(aggregated_weights, strict=True)
        self._client_weights.clear()
        self._client_samples.clear()
