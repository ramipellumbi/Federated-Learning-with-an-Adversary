# Federated-Learning-with-an-Adversary

This project implements a federated learning system with the MNIST dataset. It simulates a federated learning environment with multiple types of clients, including public, private, adversarial, and private adversarial. The project is designed to handle experiments under different settings, such as varying the number of clients, the presence of adversarial clients, and the application of differential privacy techniques.

## Overview

The system comprises a server and multiple clients. The server orchestrates the learning process across different clients, aggregates their model updates, and applies protective measures against adversarial attacks. The clients, each holding a portion of the MNIST dataset, train a local model and send their updates back to the server.

This project enables experimentation with:

- Federated learning under the influence of adversarial clients
- Differential privacy to protect client data
- The impact of dataset distribution (IID vs. non-IID) on learning

## Project Structure

- `project/`
  - `federated/`: Contains implementations for the server and client models.
  - `data_loaders/`: Data loading utilities for MNIST.
  - `main.py`: Entry point for running the experiments.
  - `models/`: Contains the neural network models used in experiments.
  - `setup.py`: Command line argument parsing and configuration setup.
  - `training.py`: Training loops for federated learning.
  - `utilities.py`: Helper functions for saving and loading results.

## Dependencies

Found in `requirements.txt`.

## Getting Started

To run the project, you will first need to install its dependencies. We recommend creating a virtual environment to manage the dependencies.

Once dependencies are installed, you can execute the project with custom configurations through command-line arguments.

### Running the Project

Navigate to the root directory and run `python project/main.py` with desired arguments, e.g., to run the project with 10 clients, including 2 adversarial, for 5 rounds with IID data, use the following command:

```bash
python project/main.py --n_clients 10 --n_adv 2 --n_rounds 5 --batch_size 64 --enable_adv_protection True --iid True
```

### Command Line Arguments

- `--n_clients`: Number of clients participating in the federated learning (default=10).
- `--n_adv`: Number of adversarial clients (default=0).
- `--noise_multiplier`: Noise multiplier for adversarial attacks (use when `n_adv` > 0).
- `--n_rounds`: Number of federated learning rounds (default=5).
- `--L`: Local epochs to train on client side before updating the server (default=-1 for all batches).
- `--batch_size`: Batch size for training (default=64).
- `--enable_adv_protection`: Enable adversarial protection on the server (default=False).
- `--iid`: Use IID distribution of data across clients (default=True).
- `--use_differential_privacy`: Enable differential privacy (default=False).
- `--eps`: Target epsilon for differential privacy (required if `use_differential_privacy` is True).
- `--delta`: Target delta for differential privacy (required if `use_differential_privacy` is True).

### Example Usage

To run the project with 10 clients, including 2 adversarial, for 5 rounds with IID data, use the following command:

```bash
python project/main.py --n_clients 10 --n_adv 2 --n_rounds 5 --iid True
```

To enable differential privacy with epsilon=1.0 and delta=1e-5, add the `--use_differential_privacy`, `--eps`, and `--delta` flags accordingly:

```bash
python project/main.py --n_clients 10 --n_adv 2 --n_rounds 5 --iid True --use_differential_privacy True --eps 1.0 --delta 1e-5
```

## Results

Results, including model performance metrics and federated learning round statistics, are saved automatically at the end of training. The `utilities.py` file includes functions for saving these results in a structured manner.
