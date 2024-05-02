import copy

import torch


def get_model_device(model: torch.nn.Module) -> torch.device:
    device = next(model.parameters()).device

    return device


def clone_module(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Create a new instance of the provided module on the device. Ensures
    the computational graph is not connected to the original module.
    """
    module_copy = copy.deepcopy(module)

    # Iterate over all parameters and buffers in the copy and move them to specified device
    for param in module_copy.parameters():
        param.data = param.data.detach().to(device=device)
        if param.grad is not None:
            param.grad = param.grad.detach().to(device=device)

    for name, buffer in module_copy.named_buffers():
        module_copy._buffers[name] = buffer.detach().to(device=device)

    return module_copy


def weight_attack(model: torch.nn.Module, noise_multiplier: float) -> torch.nn.Module:
    """
    Performs a weight attack on the given model by modifying its weights.

    This function iterates over all parameters of the provided model, and for
    those parameters that have weights, it alters the weights by adding noise.
    The noise is sampled from a normal distribution with mean 0 and standard
    deviation equal to the noise multiplier.

    Parameters:
        model (torch.nn.Module): The model whose weights will be attacked.
        noise_multiplier (float): The factor by which the weights are scaled to
                                  create noise.
    """
    # make a copy of the model
    device = get_model_device(model)
    model_copy = clone_module(model, device)

    with torch.no_grad():
        for param in model_copy.parameters():
            if param.data is not None:
                # add noise to the weights sampled from a normal distribution
                # with mean 0 and standard deviation equal to the noise multiplier
                noise = noise_multiplier * torch.normal(
                    mean=0.0, std=1, size=param.data.size(), device=device
                )
                param.data.add_(noise)

    return model_copy
