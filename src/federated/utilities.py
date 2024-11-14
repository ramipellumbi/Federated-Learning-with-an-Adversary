import os
import re
from typing import List, Tuple

import pandas as pd

from federated.federated.server import Server

from .setup import FederatedLearningConfig
from .training import TClient


def get_filenames_from_config(config: FederatedLearningConfig) -> Tuple[str, str]:
    n_clients = config.n_clients
    n_adv = config.n_adv
    # remove the .0 suffix from the float
    noise_multiplier = str(config.noise_multiplier).rstrip("0").rstrip(".")
    n_rounds = config.n_rounds
    num_internal_rounds = config.L
    batch_size = config.batch_size
    should_use_iid_training_data = config.should_use_iid_training_data
    should_enable_adv_protection = config.should_enable_adv_protection
    should_use_private_clients = config.should_use_private_clients
    target_epsilon = config.target_epsilon
    target_delta = config.target_delta
    trust_threshold = str(config.trustworthy_threshold).rstrip("0").rstrip(".")

    # Save name that has n_clients, n_adv, noise_multiplier, n_rounds, L, batch_size, should_use_iid_training_data, should_enable_adv_protection, target_epsilon, target_delta
    client_save_name = f"results/FL_Config_NClients{n_clients}_NAdv{n_adv}_NoiseMultiplier{noise_multiplier}_NRounds{n_rounds}_L{num_internal_rounds}_BatchSize{batch_size}_IID{should_use_iid_training_data}_AdvProt{should_enable_adv_protection}_PrivClients{should_use_private_clients}_Eps{target_epsilon}_Delta{target_delta}_TrustThreshold{trust_threshold}.csv"
    server_save_name = f"results/FL_Config_NClients{n_clients}_NAdv{n_adv}_NoiseMultiplier{noise_multiplier}_NRounds{n_rounds}_L{num_internal_rounds}_BatchSize{batch_size}_IID{should_use_iid_training_data}_AdvProt{should_enable_adv_protection}_PrivClients{should_use_private_clients}_Eps{target_epsilon}_Delta{target_delta}_TrustThreshold{trust_threshold}_server.csv"

    return client_save_name, server_save_name


def save_results(
    *,
    clients: List[TClient],
    server: Server,
    config: FederatedLearningConfig,
):
    # save data to csv
    all_dfs = []
    n_clients = len(clients)
    assert n_clients == config.n_clients

    for i in range(n_clients):
        all_dfs.append(clients[i].train_history)
    # combine all dataframes to one dataframe
    df = pd.concat(all_dfs)

    client_save_name, server_save_name = get_filenames_from_config(config)

    # create project/results folder if it does not exist
    os.makedirs("results", exist_ok=True)

    df.to_csv(client_save_name, index=False)
    server.val_performance.to_csv(server_save_name, index=False)


def parse_filename_to_config(filename: str) -> FederatedLearningConfig:
    # Define a regex pattern to match the components in the filename
    pattern = (
        r"FL_Config_NClients(\d+)_NAdv(\d+)_NoiseMultiplier([\d.]+)_NRounds(\d+)_L(-?\d+)_"
        r"BatchSize(\d+)_IID(.*?)_AdvProt(.*?)_PrivClients(.*?)_Eps(None|[\d.]+)_Delta(None|[\d.]+)_TrustThreshold([\d.]+)"
    )

    match = re.search(pattern, filename)

    if not match:
        # add 0 after TrustThreshold if it is missing
        w = "TrustThreshold"
        loc = filename.find(w)
        new_filename = filename[: loc + len(w)] + "0" + filename[loc + len(w) :]
        match = re.search(pattern, new_filename)
        if not match:
            raise ValueError(f"Filename format does not match expected pattern: {new_filename}")
        else:
            # TODO: maybe fix for script vs. notebook
            # rename the file to the correct format
            print(filename, new_filename)
            os.rename(os.getcwd() + "/results/" + filename, os.getcwd() + "/results/" + new_filename)

    # Extract values using the capturing groups in the regex pattern
    n_clients = int(match.group(1))
    n_adv = int(match.group(2))
    noise_multiplier = float(match.group(3))
    n_rounds = int(match.group(4))
    num_internal_rounds = int(match.group(5))
    batch_size = int(match.group(6))
    should_use_iid_training_data = match.group(7) == "True"
    should_enable_adv_protection = match.group(8) == "True"
    should_use_private_clients = match.group(9) == "True"
    target_epsilon = float(match.group(10)) if match.group(10) != "None" else 0.0
    target_delta = float(match.group(11)) if match.group(11) != "None" else 0.0
    trust_threshold = float(match.group(12).strip("."))

    # Construct and return a FederatedLearningConfig instance from values
    return FederatedLearningConfig(
        n_clients=n_clients,
        n_adv=n_adv,
        noise_multiplier=noise_multiplier,
        n_rounds=n_rounds,
        L=num_internal_rounds,
        batch_size=batch_size,
        should_use_iid_training_data=should_use_iid_training_data,
        should_enable_adv_protection=should_enable_adv_protection,
        should_use_private_clients=should_use_private_clients,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        trustworthy_threshold=trust_threshold,
    )
