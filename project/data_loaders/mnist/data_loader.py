import copy
from typing import Any, Dict, List, Union, Literal

import numpy as np
from opacus.optimizers.optimizer import Optional
from sklearn.model_selection import train_test_split
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST


def create_mislabeled_dataset(
    dataset: torch.utils.data.Subset[MNIST],
) -> torch.utils.data.Subset[MNIST]:
    """
    Modify the dataset to have mislabeled targets intentionally.
    """
    # Deepcopy to avoid modifying the original dataset

    adversarial_dataset: Any = copy.deepcopy(dataset)
    all_labels = list(set(adversarial_dataset.dataset.targets.numpy()))

    # Mislabeled strategy: Increment each label to next class modulo the number of classes
    for i in range(len(adversarial_dataset.indices)):
        # Retrieve original index to access and modify its label
        original_idx = adversarial_dataset.indices[i]
        original_label = adversarial_dataset.dataset.targets[original_idx].item()
        adversarial_label = (original_label + 1) % len(all_labels)
        adversarial_dataset.dataset.targets[original_idx] = torch.tensor(
            adversarial_label
        )

    subset = Subset(adversarial_dataset.dataset, adversarial_dataset.indices)

    return subset


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
        num_adversaries: int = 0,
    ) -> None:
        # assert ratios make sense
        assert 0.0 < val_split < 1.0
        assert 0.0 < test_split < 1.0
        assert 0.0 < val_split + test_split < 1.0
        assert 0 <= num_adversaries <= n_clients

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
                    transforms.Normalize((self.MNIST_MEAN,), (self.MNIST_STD,)),
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
                    transforms.Normalize((self.MNIST_MEAN,), (self.MNIST_STD,)),
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

        self.train_datasets = self._split_dataset_into_subsets(
            mnist_train, train_indices, self.n_clients
        )

        # set up adversarial dataset
        if num_adversaries > 0:
            for i in range(num_adversaries):
                print(f"Creating adversarial dataset {i}")
                dset = self.train_datasets[i]
                self.train_datasets[i] = create_mislabeled_dataset(dset)

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

    def _split_dataset_into_subsets(
        self, dataset: MNIST, indices: List[int], n_clients: int
    ) -> List[Subset[MNIST]]:
        # Splitting the dataset indices among N clients
        split_indices = np.array_split(indices, n_clients)
        subsets = [Subset(dataset, indices.tolist()) for indices in split_indices]
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
                collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
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
