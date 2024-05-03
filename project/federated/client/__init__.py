from typing import Union

from federated.client.private.adversarial_client import (
    AdversarialClient as PrivateAdversarialClient,
)
from federated.client.private.private_client import PrivateClient
from federated.client.public.public_client import PublicClient
from federated.client.public.adversarial_client import AdversarialClient


TClient = Union[
    PublicClient, AdversarialClient, PrivateClient, PrivateAdversarialClient
]

__all__ = [
    "PrivateAdversarialClient",
    "PrivateClient",
    "PublicClient",
    "AdversarialClient",
    "TClient",
]
