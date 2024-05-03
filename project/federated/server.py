from typing import Any, Dict, List, Union

import pandas as pd
from tqdm import tqdm
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
        weight_threshold: float,
        validation_data: torch.utils.data.DataLoader[MNIST],
        enable_adversary_protection: bool = False,
    ):
        # the server never trains, it only aggregates weights into the model
        self._model = model
        self._device = device
        self._model.to(device)

        self._validation_data = validation_data
        self._enable_adversary_protection = enable_adversary_protection
        # only weights with a trust score above this threshold will be used in
        # aggregation. If None, all weights will be used
        self._weight_threshold = weight_threshold

        # _client_weights[i] is the state dict of model trained by client i
        self._client_weights: List[Dict[str, Any]] = []
        # _client_samples_processed[i] is the #samples client i trained with
        self._client_samples: List[int] = []
        # trust scores for each client
        self._client_trust: Dict[int, float] = {}
        # used to store the weights used in an aggregation
        self._client_score: List[float] = []

        # internal validation accuracy for each aggregation round
        self._val_performance: List[Dict[str, Union[str, float]]] = []

    def _calculate_median_weights(self):
        """Calculate the median of weights across clients."""
        param_keys = list(self._client_weights[0].keys())
        median_weights = {}

        # For each parameter, stack across clients and calculate median
        for key in param_keys:
            # move to CPU because MPS does not support >4 dimensional tensors
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
        Compute the trust scores for each client based on the deviation of
        their weights from the median.
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
            for images, labels in tqdm(self._validation_data, desc="Validating model"):
                outputs = self._model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        print(f"Accuracy of the server on the validation images: {accuracy:.2f}%")

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
        Add the parameters of a client to the server and the number of samples
        it used for training.
        """
        client_id = len(self._client_weights)
        self._client_weights.append(client_state_dict)
        self._client_samples.append(num_samples)

        # Initialize trust score uniformly among all known clients
        if client_id not in self._client_trust:
            # Reset all to equal trust
            for i in range(len(self._client_weights)):
                self._client_trust[i] = 1.0 / len(self._client_weights)

    def aggregate_parameters(self, is_verbose: bool):
        """
        Aggregate the parameters of all clients and update the model with the
        aggregated weights.

        If adversary_protection is True, the server will compute trust scores
        for each client and reweight their contributions to the aggregation.
        """
        self._model.train()

        # if we are using adversary protection, compute trust scores to
        # reweight client contributions
        if self._enable_adversary_protection:
            self._compute_trust_scores()

        total_samples = sum(self._client_samples)

        # this is the ratio of data client i has compared to all clients
        # NOTE: sums to 1
        client_ratios = [
            self._client_samples[i] / total_samples
            for i in range(len(self._client_samples))
        ]

        # only use clients with trust scores above the threshold
        weight_indices_to_use = [
            i
            for i in range(len(self._client_weights))
            if self._client_trust[i] > self._weight_threshold
        ]
        if is_verbose:
            print(
                f"Using {len(weight_indices_to_use)} clients out of {len(self._client_weights)}"
            )
            print(f"Clients to use: {weight_indices_to_use}")

        # this is the ratio of data client i has compared to all clients
        # weighted by trust -- NOTE: does NOT sum to 1 necessarily
        new_contribution = [
            self._client_trust[i] * client_ratios[i] for i in weight_indices_to_use
        ]

        # normalize contributions to sum to 1
        total_new_contribution = sum(new_contribution)
        client_contributions = [
            contribution / total_new_contribution for contribution in new_contribution
        ]
        if is_verbose:
            print(f"Client contributions: {client_contributions}")

        # store the client scores for next round weighting
        self._client_score = client_contributions.copy()

        client_weights_to_use = [self._client_weights[i] for i in weight_indices_to_use]

        # aggregate the weights into the server model
        aggregated_weights = {
            name: torch.zeros_like(param, device=self._device)
            for name, param in self._model.named_parameters()
        }
        with torch.no_grad():
            for client_weight, weight_fraction in zip(
                client_weights_to_use, client_contributions
            ):
                for name, param in client_weight.items():
                    # dp waits are prepended with _module
                    name = name.replace("_module.", "")
                    differential = param.clone().detach() * weight_fraction
                    aggregated_weights[name] += differential

        self._model.load_state_dict(aggregated_weights, strict=True)
        self._client_weights.clear()
        self._client_samples.clear()
