import matplotlib.pyplot as plt
import numpy as np
from torch import device

from data_loader import DataLoader


if __name__ == "__main__":
    data_loader = DataLoader(batch_size=64, device=device("mps"), n_clients=10)

    # plot the label distribution for the training set, validation set, and test set

    label_counts = data_loader.get_label_distribution("train")
    if isinstance(label_counts, list):
        for i, label_count in enumerate(label_counts):
            labels, counts = zip(*label_count.items())
            x = np.arange(len(labels))
            y = np.array(counts)
            # histogram
            plt.bar(x, y)
            plt.savefig(f"data_distribution/client_{i}_train.png")
            plt.close()

    label_counts = data_loader.get_label_distribution("val")
    if isinstance(label_counts, dict):
        labels, counts = zip(*label_counts.items())
        x = np.arange(len(labels))
        y = np.array(counts)
        # histogram
        plt.bar(x, y)
        plt.savefig("data_distribution/val.png")
        plt.close()

    label_counts = data_loader.get_label_distribution("test")
    if isinstance(label_counts, dict):
        labels, counts = zip(*label_counts.items())
        x = np.arange(len(labels))
        y = np.array(counts)
        # histogram
        plt.bar(x, y)
        plt.savefig("data_distribution/test.png")
        plt.close()

    adversarial_data_loader = DataLoader(
        batch_size=64, device=device("mps"), n_clients=10, num_adversaries=1
    )
    # plot the label distribution for the adversarial set
    adv_loader = adversarial_data_loader._train_loaders[0]

    # plot some samples and their labels
    plt.figure(figsize=(12, 12))
    for i, (data, labels) in enumerate(adv_loader):
        if i == 10:
            break
        for j in range(10):
            plt.subplot(10, 10, i * 10 + j + 1)
            plt.imshow(data[j].squeeze().cpu().numpy(), cmap="gray")
            plt.axis("off")
            plt.title(f"Label: {labels[j].item()}")

    plt.show()
