import torch


def gradient_attack(model: torch.nn.Module, noise_multiplier: float):
    """
    Performs a gradient attack on the given model by modifying its gradients.

    This function iterates over all parameters of the provided model, and for
    those parameters that have gradients, it alters the gradients by adding noise.
    The noise is computed as the product of a specified noise multiplier and the
    negative sign of the gradient. This approach aims to perturb the gradient
    descent process, potentially leading to adversarial effects on the model's
    training.

    Parameters:
        model (torch.nn.Module): The model whose gradients will be attacked.
        noise_multiplier (float): The factor by which the gradient sign is scaled to
                                  create noise.

    Returns:
        None: The function modifies the model's gradients in-place and does not return
              any value.
    """
    # copy the model
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                # Instead of altering input, the gradient is modified
                param.grad += noise_multiplier * -1 * torch.sign(param.grad)
