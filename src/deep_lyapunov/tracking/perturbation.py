"""Perturbation utilities for multi-trajectory analysis."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

# Module-level logger
logger = logging.getLogger(__name__)


def apply_perturbation(
    model: nn.Module,
    scale: float = 0.01,
    seed: Optional[int] = None,
) -> None:
    """Apply Gaussian perturbation to model parameters in-place.

    Adds small random noise to each parameter, scaled by the parameter's
    standard deviation. This is useful for creating perturbed copies of
    a model to analyze training stability.

    Args:
        model: PyTorch model to perturb. Modified in-place.
        scale: Perturbation magnitude as fraction of parameter std.
            Default 0.01 (1%).
        seed: Random seed for reproducibility. If None, uses current
            random state.

    Example:
        >>> model = nn.Linear(10, 5)
        >>> apply_perturbation(model, scale=0.01, seed=42)
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                std = param.std().item()
                # Use small default if parameter has zero variance
                if std <= 0:
                    std = 0.01
                noise = torch.randn_like(param) * scale * std
                param.add_(noise)


def create_perturbed_copies(
    model: nn.Module,
    n_copies: int,
    scale: float = 0.01,
    base_seed: int = 42,
    include_original: bool = True,
) -> List[nn.Module]:
    """Create multiple perturbed copies of a model.

    Creates N copies of the model with small Gaussian perturbations applied
    to their parameters. This is the core setup for perturbation-based
    stability analysis.

    Args:
        model: Base PyTorch model to copy and perturb.
        n_copies: Number of perturbed copies to create.
        scale: Perturbation magnitude as fraction of parameter std.
        base_seed: Starting seed for reproducible perturbations.
        include_original: If True, first copy is unperturbed (the original).

    Returns:
        List of n_copies models. If include_original is True, the first
        model is unperturbed.

    Example:
        >>> base_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> models = create_perturbed_copies(base_model, n_copies=5)
        >>> len(models)
        5
        >>> # First is unperturbed, rest have small perturbations
    """
    import torch

    logger.debug(
        f"Creating {n_copies} perturbed copies (scale={scale}, "
        f"include_original={include_original})"
    )

    # Get base state
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    copies = []

    for i in range(n_copies):
        # Deep copy the model architecture
        model_copy = copy.deepcopy(model)
        model_copy.load_state_dict({k: v.clone() for k, v in base_state.items()})

        # Apply perturbation to all but the first (if include_original)
        if not include_original or i > 0:
            seed = base_seed + i
            apply_perturbation(model_copy, scale=scale, seed=seed)
            logger.debug(f"  Copy {i}: perturbed with seed={seed}")
        else:
            logger.debug(f"  Copy {i}: unperturbed (original)")

        copies.append(model_copy)

    logger.debug(f"Created {len(copies)} model copies")
    return copies


def compute_perturbation_magnitude(
    original: nn.Module,
    perturbed: nn.Module,
) -> float:
    """Compute the L2 distance between two models' parameters.

    Useful for verifying perturbation scale or measuring how far
    models have diverged during training.

    Args:
        original: First model.
        perturbed: Second model (typically a perturbed copy).

    Returns:
        L2 distance between the flattened parameter vectors.

    Example:
        >>> model1 = nn.Linear(10, 5)
        >>> model2 = copy.deepcopy(model1)
        >>> apply_perturbation(model2, scale=0.01)
        >>> dist = compute_perturbation_magnitude(model1, model2)
        >>> print(f"Distance: {dist:.6f}")
    """
    import torch

    distance_sq = 0.0

    for p1, p2 in zip(original.parameters(), perturbed.parameters()):
        diff = p1.detach() - p2.detach()
        distance_sq += torch.sum(diff**2).item()

    return float(distance_sq**0.5)


def get_parameter_statistics(model: nn.Module) -> dict:
    """Get statistics about model parameters.

    Useful for understanding the scale of parameters when choosing
    perturbation magnitude.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with parameter statistics:
        - n_parameters: Total trainable parameters
        - mean: Mean of all parameters
        - std: Standard deviation of all parameters
        - min: Minimum parameter value
        - max: Maximum parameter value
    """
    import torch

    all_params = []
    for param in model.parameters():
        if param.requires_grad:
            all_params.append(param.detach().cpu().flatten())

    if not all_params:
        return {
            "n_parameters": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    flat = torch.cat(all_params)
    return {
        "n_parameters": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }
