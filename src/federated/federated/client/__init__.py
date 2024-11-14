from typing import Union

from .private.adversarial_client import (
    AdversarialClient as PrivateAdversarialClient,
)
from .private.private_client import PrivateClient
from .public.adversarial_client import AdversarialClient
from .public.public_client import PublicClient

TClient = Union[PublicClient, AdversarialClient, PrivateClient, PrivateAdversarialClient]

__all__ = [
    "PrivateAdversarialClient",
    "PrivateClient",
    "PublicClient",
    "AdversarialClient",
    "TClient",
]
