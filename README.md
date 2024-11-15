# Byzantine Robust Federated Learning via Client Trustworthiness Scores

This project implements a federated learning system with the MNIST dataset. It simulates a federated learning environment with multiple types of clients, including public, private, adversarial, and private adversarial. The project is designed to handle experiments under different settings, such as varying the number of clients, the presence of adversarial clients, and the application of differential privacy techniques.

## Overview

The system comprises a server and multiple clients. The server orchestrates the learning process across different clients, aggregates their model updates, and applies protective measures against adversarial attacks. The clients, each holding a portion of the MNIST dataset, train a local model and send their updates back to the server.

This project enables experimentation with:

- Federated learning under the influence of adversarial clients
- Differential privacy to protect client data
- The impact of dataset distribution (IID vs. non-IID) on learning

For a full description of the problem and experimental results, refer to the [report](./report.pdf).

## Project Files

- `src/federated/`
  - `attacks/`: Contains adversarial attacks on the weights a
    client sends to the server.
  - `data_loaders/`: Data loading utilities for MNIST.
  - `federated/`: Contains implementations for the server and client models.
    - `client/`: Contains the implementation of a client, differentially private client, adversarial client, and
      differentially private adversarial client.
    - `server.py`: Contains the implementation of the server, which aggregates the weights from clients after a communication round.
  - `models/`: Contains the neural network models used in experiments.
  - `results/`: Contains the results of the experiments.
  - `main.py`: Entry point for running the experiments.
  - `setup.py`: Command line argument parsing, validation, and configuration.
  - `training.py`: Training and evaluation loops for federated learning.
  - `utilities.py`: Helper functions for saving and loading results.

## Dependencies

Found in [`pyproject.toml`](./pyproject.toml). This project used `pdm`.

## Getting Started

To run the project, you will first need to install its dependencies. We recommend creating a virtual environment to manage the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then, install the dependencies:

```bash
pdm install
```

Once dependencies are installed, you can execute the project with custom configurations through command-line arguments.

### Running the Project

Navigate to the root directory and run `python -m federated.main` with desired arguments, e.g., to run the project with 10 clients, including 2 adversarial, for 5 rounds with IID data, use the following command:

```bash
python -m federated.main --n_clients 10 --n_adv 2 --noise_multiplier=0.1 --n_rounds 5 --batch_size 64 --enable_adv_protection True --iid True
```

- If you would prefer to use a Jupyter notebook, you can run the `project/run_single_experiment.ipynb` notebook instead, which
  is set up to run one experiment at a time.
- To perform multiple experiments, you can run the notebook `project/run_multiple_experiments.ipynb`, which is set up to
  perform multiple experiments with different configurations.
- To evaluate / plot the results of experiments, you can run the notebook `project/evaluate.ipynb`.

### Command Line Arguments

- `--n_clients`: Number of clients participating in the federated learning (default=10).
- `--n_adv`: Number of adversarial clients (default=0).
- `--noise_multiplier`: Noise multiplier for adversarial attacks (required when `n_adv` > 0).
- `--n_rounds`: Number of federated learning rounds (default=5).
- `--L`: Local epochs to train on client side before updating the server (default=-1 for all batches).
- `--batch_size`: Batch size for training (default=64).
- `--enable_adv_protection`: Enable adversarial protection on the server (default=False).
- `--trust_threshold`: Threshold for a client's trustworthiness score in order for their weights to be a part of the aggregation. (default=0.0, i.e., use all weights).
- `--iid`: Use IID distribution of data across clients (default=True).
- `--use_differential_privacy`: Enable differential privacy (default=False).
- `--eps`: Target epsilon for differential privacy (required if `use_differential_privacy` is True).
- `--delta`: Target delta for differential privacy (required if `use_differential_privacy` is True).

### Example Usage

To run the project with 10 clients, including 2 adversarial, for 5 rounds with IID data, use the following command:

```bash
python -m federated.main --n_clients 10 --n_adv 2 --noise_multiplier 0.1 --n_rounds 5 --iid True
```

To enable differential privacy with epsilon=1.0 and delta=1e-5, add the `--use_differential_privacy`, `--eps`, and `--delta` flags accordingly:

```bash
python -m federated.main --n_clients 10 --n_adv 2 --noise_multiplier 0.1 --n_rounds 5 --iid True --use_differential_privacy True --eps 1.0 --delta 1e-5
```

## Results

Results, including model performance metrics and federated learning round statistics, are saved automatically at the end of training in the `src/federated/results/` directory. The `utilities.py` file includes functions for saving and parsing these results in a structured manner.
