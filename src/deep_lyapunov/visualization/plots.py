"""Plotting functions for stability analysis visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure


def plot_trajectories_2d(
    pca_trajectories: np.ndarray,
    pca_variance: np.ndarray,
    ax: Optional[plt.Axes] = None,
    components: Tuple[int, int] = (0, 1),
    colormap: str = "viridis",
    alpha: float = 0.7,
    show_endpoints: bool = True,
) -> plt.Figure:
    """Plot weight trajectories in 2D PCA space.

    Args:
        pca_trajectories: Array of shape (n_checkpoints, n_trajectories, n_components).
        pca_variance: Explained variance ratio for each component.
        ax: Matplotlib axes. If None, creates new figure.
        components: Which PCA components to plot (default: first two).
        colormap: Matplotlib colormap name.
        alpha: Line transparency.
        show_endpoints: Whether to mark start/end points.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    n_checkpoints, n_trajectories, _ = pca_trajectories.shape
    c1, c2 = components

    colors = plt.colormaps.get_cmap(colormap)(np.linspace(0, 1, n_trajectories))

    for i in range(n_trajectories):
        traj = pca_trajectories[:, i, :]
        ax.plot(
            traj[:, c1],
            traj[:, c2],
            color=colors[i],
            alpha=alpha,
            linewidth=1.5,
            label=f"Trajectory {i+1}",
        )

        if show_endpoints:
            # Start point (circle)
            ax.scatter(
                traj[0, c1],
                traj[0, c2],
                color=colors[i],
                s=100,
                marker="o",
                edgecolors="black",
                zorder=5,
            )
            # End point (square)
            ax.scatter(
                traj[-1, c1],
                traj[-1, c2],
                color=colors[i],
                s=100,
                marker="s",
                edgecolors="black",
                zorder=5,
            )

    # Labels with variance info
    var1 = pca_variance[c1] * 100 if c1 < len(pca_variance) else 0
    var2 = pca_variance[c2] * 100 if c2 < len(pca_variance) else 0
    ax.set_xlabel(f"PC{c1+1} ({var1:.1f}% var)")
    ax.set_ylabel(f"PC{c2+1} ({var2:.1f}% var)")
    ax.set_title("Weight Trajectories in PCA Space")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig


def plot_trajectories_3d(
    pca_trajectories: np.ndarray,
    pca_variance: np.ndarray,
    ax: Optional[plt.Axes] = None,
    components: Tuple[int, int, int] = (0, 1, 2),
    colormap: str = "viridis",
    alpha: float = 0.7,
) -> plt.Figure:
    """Plot weight trajectories in 3D PCA space.

    Args:
        pca_trajectories: Array of shape (n_checkpoints, n_trajectories, n_components).
        pca_variance: Explained variance ratio for each component.
        ax: Matplotlib 3D axes. If None, creates new figure.
        components: Which PCA components to plot (default: first three).
        colormap: Matplotlib colormap name.
        alpha: Line transparency.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    n_checkpoints, n_trajectories, _ = pca_trajectories.shape
    c1, c2, c3 = components

    colors = plt.colormaps.get_cmap(colormap)(np.linspace(0, 1, n_trajectories))

    for i in range(n_trajectories):
        traj = pca_trajectories[:, i, :]
        ax.plot(
            traj[:, c1],
            traj[:, c2],
            traj[:, c3],
            color=colors[i],
            alpha=alpha,
            linewidth=1.5,
        )
        # Start and end markers
        ax.scatter(
            [traj[0, c1]],
            [traj[0, c2]],
            [traj[0, c3]],
            color=colors[i],
            s=100,
            marker="o",
        )
        ax.scatter(
            [traj[-1, c1]],
            [traj[-1, c2]],
            [traj[-1, c3]],
            color=colors[i],
            s=100,
            marker="s",
        )

    ax.set_xlabel(f"PC{c1+1}")
    ax.set_ylabel(f"PC{c2+1}")
    ax.set_zlabel(f"PC{c3+1}")
    ax.set_title("3D Weight Trajectory")

    return fig


def plot_spread_evolution(
    spread: np.ndarray,
    ax: Optional[plt.Axes] = None,
    behavior: str = "unknown",
    convergence_ratio: Optional[float] = None,
) -> plt.Figure:
    """Plot trajectory spread evolution over training.

    Args:
        spread: Array of spread values at each checkpoint.
        ax: Matplotlib axes. If None, creates new figure.
        behavior: "convergent" or "divergent" for coloring.
        convergence_ratio: Optional ratio to display in title.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    epochs = np.arange(len(spread))
    color = "forestgreen" if behavior == "convergent" else "crimson"

    ax.plot(epochs, spread, color=color, linewidth=2)
    ax.axhline(
        y=spread[0], color="gray", linestyle="--", alpha=0.7, label="Initial spread"
    )
    ax.fill_between(epochs, spread, spread[0], alpha=0.2, color=color)

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Trajectory Spread")

    title = f"Spread Evolution ({behavior.title()})"
    if convergence_ratio is not None:
        title += f"\nRatio: {convergence_ratio:.2f}x"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_lyapunov_evolution(
    spread: np.ndarray,
    global_lyapunov: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot local Lyapunov exponents over training.

    Args:
        spread: Array of spread values at each checkpoint.
        global_lyapunov: Overall Lyapunov exponent for annotation.
        ax: Matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if len(spread) < 2:
        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Compute local Lyapunov
    local_lyap = np.log(spread[1:] / np.maximum(spread[:-1], 1e-10))
    epochs = np.arange(1, len(spread))

    # Bar plot colored by sign
    colors = ["forestgreen" if l < 0 else "crimson" for l in local_lyap]
    ax.bar(epochs, local_lyap, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.axhline(
        y=global_lyapunov,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean λ = {global_lyapunov:.4f}",
    )

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Local Lyapunov Exponent")
    ax.set_title("Lyapunov Exponent Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_convergence_basin(
    pca_trajectories: np.ndarray,
    pca_variance: np.ndarray,
    ax: Optional[plt.Axes] = None,
    components: Tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Plot initial and final point distributions.

    Shows how the basin of attraction changes during training.

    Args:
        pca_trajectories: Array of shape (n_checkpoints, n_trajectories, n_components).
        pca_variance: Explained variance ratio for each component.
        ax: Matplotlib axes. If None, creates new figure.
        components: Which PCA components to plot.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    c1, c2 = components

    # Initial points
    initial = pca_trajectories[0, :, :]
    ax.scatter(
        initial[:, c1],
        initial[:, c2],
        c="green",
        s=200,
        marker="o",
        alpha=0.7,
        edgecolors="black",
        label="Initial points",
    )

    # Final points
    final = pca_trajectories[-1, :, :]
    ax.scatter(
        final[:, c1],
        final[:, c2],
        c="red",
        s=200,
        marker="*",
        alpha=0.7,
        edgecolors="black",
        label="Final points",
    )

    # Draw convex hulls if enough points
    if len(initial) >= 3:
        try:
            hull = ConvexHull(initial[:, [c1, c2]])
            for simplex in hull.simplices:
                ax.plot(initial[simplex, c1], initial[simplex, c2], "g--", alpha=0.5)
        except Exception:
            pass

    if len(final) >= 3:
        try:
            hull = ConvexHull(final[:, [c1, c2]])
            for simplex in hull.simplices:
                ax.plot(final[simplex, c1], final[simplex, c2], "r--", alpha=0.5)
        except Exception:
            pass

    # Compute spreads for title
    init_spread = np.std(initial[:, [c1, c2]])
    final_spread = np.std(final[:, [c1, c2]])

    ax.set_xlabel(f"PC{c1+1}")
    ax.set_ylabel(f"PC{c2+1}")
    ax.set_title(
        f"Convergence Basin\nInit spread: {init_spread:.3f}, Final spread: {final_spread:.3f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def create_analysis_dashboard(
    pca_trajectories: np.ndarray,
    pca_variance: np.ndarray,
    spread_evolution: np.ndarray,
    lyapunov: float,
    convergence_ratio: float,
    behavior: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create comprehensive analysis dashboard.

    Args:
        pca_trajectories: Projected trajectories.
        pca_variance: PCA explained variance ratios.
        spread_evolution: Spread at each checkpoint.
        lyapunov: Global Lyapunov exponent.
        convergence_ratio: Convergence ratio.
        behavior: "convergent" or "divergent".
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure with 2x2 subplot grid.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: 2D trajectories
    plot_trajectories_2d(pca_trajectories, pca_variance, ax=axes[0, 0])

    # Top-right: Convergence basin
    plot_convergence_basin(pca_trajectories, pca_variance, ax=axes[0, 1])

    # Bottom-left: Spread evolution
    plot_spread_evolution(
        spread_evolution,
        ax=axes[1, 0],
        behavior=behavior,
        convergence_ratio=convergence_ratio,
    )

    # Bottom-right: Lyapunov evolution
    plot_lyapunov_evolution(spread_evolution, lyapunov, ax=axes[1, 1])

    plt.suptitle(
        f"Stability Analysis Dashboard\n"
        f"Convergence: {convergence_ratio:.2f}x | Lyapunov: {lyapunov:.4f} | {behavior.title()}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
