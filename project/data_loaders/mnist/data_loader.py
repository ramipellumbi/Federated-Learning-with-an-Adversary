from collections import defaultdict
from typing import Dict, List, Union, Literal

import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST


class DataLoader:
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081

    def __init__(
        self,
        *,
        batch_size: int,
        device: torch.device,
        val_split: float = 0.1,
        test_split: float = 0.1,
        n_clients: int = 1,
        use_iid: bool = True,
        # num_adversaries: int = 0,
    ) -> None:
        # assert ratios make sense
        assert 0.0 < val_split < 1.0
        assert 0.0 < test_split < 1.0
        assert 0.0 < val_split + test_split < 1.0
        # assert 0 <= num_adversaries <= n_clients

        self.batch_size = batch_size
        self.device = device
        self.val_split = val_split
        self.test_split = test_split
        self.n_clients = n_clients

        # Download the MNIST dataset
        mnist_train = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((self.MNIST_MEAN,),
                                         (self.MNIST_STD,)),
                ]
            ),
        )

        mnist_test = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((self.MNIST_MEAN,),
                                         (self.MNIST_STD,)),
                ]
            ),
        )

        # Splitting dataset into train, validation, and test sets
        indices = list(range(len(mnist_train)))
        train_indices, val_test_indices = train_test_split(
            indices,
            test_size=(self.val_split + self.test_split),
            random_state=42,
            shuffle=True,
        )
        val_indices, test_indices = train_test_split(
            val_test_indices,
            test_size=self.test_split / (self.val_split + self.test_split),
            random_state=42,
            shuffle=True,
        )

        if use_iid:
            self.train_datasets = self._split_dataset_into_subsets_iid(
                mnist_train, train_indices, self.n_clients
            )
        else:
            self.train_datasets = self._split_dataset_into_subsets_non_iid(
                mnist_train, train_indices, self.n_clients
            )

        self.val_dataset: Subset[MNIST] = Subset(mnist_train, val_indices)
        self.test_dataset: Union[MNIST, Subset[MNIST]] = (
            Subset(mnist_train, test_indices) if len(test_indices) > 0 else mnist_test
        )

        self._train_loaders = self._create_data_loaders(
            self.train_datasets, self.batch_size, device
        )
        self._val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
            pin_memory=True,
        )
        self._test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
            pin_memory=True,
        )

    def _split_dataset_into_subsets_non_iid(
        self, dataset: MNIST, indices: List[int], n_clients: int
    ) -> List[Subset[MNIST]]:
        # Initialize a dictionary to keep track of indices for each label.
        label_to_indices = defaultdict(list)
        for idx in indices:
            label = dataset.targets[idx].item()
            label_to_indices[label].append(idx)

        # Pre-allocate minimum samples per label for each client
        min_samples_per_label = 30
        client_indices = [list() for _ in range(n_clients)]

        # Allocate at least 30 samples of each label to every client
        for label, indices in label_to_indices.items():
            np.random.shuffle(indices)
            allocated_indices = 0
            for client_id in range(n_clients):
                client_specific_indices = indices[
                    allocated_indices: allocated_indices + min_samples_per_label
                ]
                client_indices[client_id].extend(client_specific_indices)
                allocated_indices += min_samples_per_label

                # Break if we've allocated min samples and cannot fulfill
                # another full round
                if allocated_indices + min_samples_per_label > len(indices):
                    break

        # Allocate remaining data in a non-i.i.d fashion
        for label, indices in label_to_indices.items():
            remaining_indices = indices[
                allocated_indices:
            ]  # Get remaining indices after min allocation
            np.random.shuffle(remaining_indices)

            # Generate proportions with a strong skew for uneven distribution
            # of the remaining data
            alpha = np.random.random(n_clients) * 10
            proportions_remaining = np.random.dirichlet(alpha)

            # Calculate end indices based on proportions for remaining data
            n_remaining_indices = len(remaining_indices)
            proportions_end_index = [
                int(proportion * n_remaining_indices)
                for proportion in proportions_remaining
            ]

            start_idx = 0
            for client_id, end_idx_increment in enumerate(proportions_end_index):
                end_idx = start_idx + end_idx_increment
                client_specific_remaining_indices = remaining_indices[start_idx:end_idx]
                client_indices[client_id].extend(client_specific_remaining_indices)
                start_idx = end_idx

        # Shuffle the indices within each client to ensure randomness
        for i in range(n_clients):
            np.random.shuffle(client_indices[i])

        subsets = [Subset(dataset, indices) for indices in client_indices]
        return subsets

    def _split_dataset_into_subsets_iid(
        self, dataset: MNIST, indices: List[int], n_clients: int
    ) -> List[Subset[MNIST]]:
        # Splitting the dataset indices among N clients
        split_indices = np.array_split(indices, n_clients)
        subsets = [Subset(dataset, indices.tolist())
                   for indices in split_indices]
        return subsets

    def _create_data_loaders(
        self, datasets: List[Subset[MNIST]], batch_size: int, device: torch.device
    ) -> List[torch.utils.data.DataLoader[MNIST]]:
        # Creating a DataLoader for each subset
        loaders: List[torch.utils.data.DataLoader[MNIST]] = []
        for dataset in datasets:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=lambda x: tuple(x_.to(device)
                                           for x_ in default_collate(x)),
                pin_memory=True,
            )
            loaders.append(loader)
        return loaders

    @property
    def train_loaders(self):
        return self._train_loaders

    @property
    def val_loader(self):
        return self._val_loader

    @property
    def test_loader(self):
        return self._test_loader

    def get_label_distribution(
        self, type: Union[Literal["train"], Literal["val"], Literal["test"]]
    ) -> Union[List[Dict[int, int]], Dict[int, int]]:
        """
        For each label, get the count of the number of samples in the dataset.
        """
        assert type in [
            "train",
            "val",
            "test",
        ], "type should be either 'train', 'val', or 'test'"
        if type == "train":
            loaders = self._train_loaders
        else:
            loader = self._val_loader if type == "val" else self._test_loader
            # Wrap the single loader in a list for uniform processing
            loaders = [loader]

        label_counts = []
        for loader in loaders:
            label_count = {}
            for _, labels in loader:
                for label in labels:
                    label = label.item()
                    if label in label_count:
                        label_count[label] += 1
                    else:
                        label_count[label] = 1
            label_counts.append(label_count)

        # return a list, where each index i is the distribution of client i
        if type == "train":
            return label_counts

        # return a single dictionary for the validation and test set (
        # distribution of that set)
        return label_counts[0]
