import matplotlib.pyplot as plt
import numpy as np
from torch import device

from data_loader import DataLoader


if __name__ == "__main__":
    use_iid = False
    data_loader = DataLoader(
        batch_size=64, device=device("mps"), n_clients=10, use_iid=use_iid
    )

    # plot the label distribution for the training, validation, and test set
    label_counts = data_loader.get_label_distribution("train")
    if isinstance(label_counts, list):
        for i, label_count in enumerate(label_counts):
            labels, counts = zip(*label_count.items())
            x = np.arange(len(labels))
            y = np.array(counts)
            # histogram
            plt.bar(x, y)
            plt.savefig(f"data_distribution/client_{i}_{use_iid}_train.png")
            plt.close()

    label_counts = data_loader.get_label_distribution("val")
    if isinstance(label_counts, dict):
        labels, counts = zip(*label_counts.items())
        x = np.arange(len(labels))
        y = np.array(counts)
        # histogram
        plt.bar(x, y)
        plt.savefig(f"data_distribution/val_{use_iid}.png")
        plt.close()

    label_counts = data_loader.get_label_distribution("test")
    if isinstance(label_counts, dict):
        labels, counts = zip(*label_counts.items())
        x = np.arange(len(labels))
        y = np.array(counts)
        # histogram
        plt.bar(x, y)
        plt.savefig(f"data_distribution/test_{use_iid}.png")
        plt.close()
