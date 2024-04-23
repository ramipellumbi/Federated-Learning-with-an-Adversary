import argparse
import dataclasses


@dataclasses.dataclass
class FederatedLearningConfig:
    n_clients: int
    n_adv: int
    noise_multiplier: float
    n_rounds: int
    L: int
    batch_size: int
    should_use_iid_training_data: bool
    should_enable_adv_protection: bool
    should_use_private_clients: bool
    target_epsilon: float
    target_delta: float


def validate_command_line_arguments(args):
    """
    Validates the command line arguments provided for configuring the federated learning system.

    This function asserts various conditions to ensure the provided arguments are valid. These conditions include:
    - The number of clients must be a positive integer.
    - The number of adversaries must be a non-negative integer and less than the number of clients.
    - If adversaries are used, a noise multiplier must be provided, be a float, and greater than 0.
    - The number of communication rounds (epochs) must be a positive integer.
    - The number of local batches, if provided, must be a positive integer.
    - The batch size must be a positive integer.
    - The flags for using differential privacy, iid data distribution, and adversarial protection must be booleans.
    - If differential privacy is enabled, epsilon and delta must be provided and be positive floats.

    Parameters:
        args (argparse.Namespace): The command line arguments parsed from the user input.

    Raises:
        AssertionError: If any of the validation conditions are not met.
        ValueError: If differential privacy is enabled but epsilon or delta are not provided or invalid.
    """
    # assert at least one client and that the client number is an integer
    assert args.n_clients > 0, "Number of clients must be greater than 0"
    assert isinstance(args.n_clients, int), "Number of clients must be an integer"

    # assert that the number of adversaries is a non-negative integer and less than the number of clients
    assert args.n_adv >= 0, "Number of adversaries must be non-negative"
    assert isinstance(args.n_adv, int), "Number of adversaries must be an integer"
    assert (
        args.n_adv < args.n_clients
    ), "Number of adversaries must be less than number of clients"

    # if the number of adversaries is greater than 0, assert that the noise multiplier is provided and greater than 0
    if args.n_adv > 0:
        assert (
            args.noise_multiplier != -1
        ), "Noise multiplier must be provided when using adversaries"
        assert isinstance(
            args.noise_multiplier, float
        ), "Noise multiplier must be a float"
        assert args.noise_multiplier > 0, "Noise multiplier must be greater than 0"

    # assert that the number of epochs is a positive integer
    assert args.n_rounds > 0, "Number of rounds must be greater than 0"
    assert isinstance(args.n_rounds, int), "Number of epochs must be an integer"
    assert isinstance(args.L, int), "Number of local batches must be an integer"
    assert args.L == -1 or args.L > 0, "Number of local batches must be greater than 0"

    # assert that the batch size is a positive integer
    assert args.batch_size > 0, "Batch size must be greater than 0"
    assert isinstance(args.batch_size, int), "Batch size must be an integer"

    # assert that use differential privacy is a boolean
    assert isinstance(
        args.use_differential_privacy, bool
    ), "Use differential privacy must be a boolean"

    # if differential privacy is enabled, assert that epsilon and delta are provided
    if args.use_differential_privacy:
        if args.eps == -1 or args.delta == -1:
            raise ValueError(
                "If you want to use private clients, you must provide epsilon and delta"
            )

        assert args.eps > 0, "Epsilon must be greater than 0"
        assert isinstance(args.eps, float), "Epsilon must be a float"

        assert args.delta > 0, "Delta must be greater than 0"
        assert isinstance(args.delta, float), "Delta must be a float"

    # assert that the iid flag is a boolean
    assert isinstance(args.iid, bool), "IID must be a boolean"

    # assert that the enable adv protection flag is a boolean
    assert isinstance(
        args.enable_adv_protection, bool
    ), "Enable adv protection must be a boolean"


def get_command_line_args():
    """
    Parses command line arguments for configuring the federated learning system.

    Returns:
        FederatedLearningConfig: A dataclass object containing the configuration parameters for the federated learning system.
    """
    # get inputs from argparse
    parser = argparse.ArgumentParser()

    # number of clients in the federated learning system
    parser.add_argument("--n_clients", type=int, default=10)
    # number of adversarial clients in the federated learning system
    parser.add_argument("--n_adv", type=int, default=0)
    # noise delta for adversarial client weights
    parser.add_argument("--noise_multiplier", type=float, default=-1)
    # number of communication rounds
    parser.add_argument("--n_rounds", type=int, default=5)
    # number of local batches. -1 means each client trains on all their batches in a communication round
    parser.add_argument("--L", type=int, default=-1)
    # enable adversarial protection on the server
    parser.add_argument("--enable_adv_protection", type=bool, default=False)
    # enable iid data distribution
    parser.add_argument("--iid", type=bool, default=True)
    # batch size a client trains on
    parser.add_argument("--batch_size", type=int, default=64)
    # enable differential privacy
    parser.add_argument("--use_differential_privacy", type=bool, default=False)
    # epsilon for differential privacy. -1 means epsilon is not provided
    parser.add_argument("--eps", type=float, default=-1)
    # delta for differential privacy. -1 means delta is not provided
    parser.add_argument("--delta", type=float, default=-1)
    args = parser.parse_args()

    # validate the command line arguments
    validate_command_line_arguments(args)

    return FederatedLearningConfig(
        n_clients=args.n_clients,
        n_adv=args.n_adv,
        noise_multiplier=args.noise_multiplier,
        n_rounds=args.n_rounds,
        L=args.L,
        batch_size=args.batch_size,
        should_use_iid_training_data=args.iid,
        should_enable_adv_protection=args.enable_adv_protection,
        should_use_private_clients=args.use_differential_privacy,
        target_epsilon=args.eps,
        target_delta=args.delta,
    )
