"""Results dataclasses for deep-lyapunov analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure


@dataclass
class TrajectoryMetrics:
    """Metrics for a single trajectory.

    Attributes:
        trajectory_id: Identifier for this trajectory.
        final_loss: Final training loss.
        final_accuracy: Final accuracy (if applicable).
        path_length: Total distance traveled in weight space.
        velocity_mean: Mean velocity (weight change rate).
        velocity_std: Standard deviation of velocity.
    """

    trajectory_id: int
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    path_length: float = 0.0
    velocity_mean: float = 0.0
    velocity_std: float = 0.0


@dataclass
class AnalysisResults:
    """Complete results from stability analysis.

    This class holds all computed metrics and provides methods for
    visualization and export.

    Attributes:
        convergence_ratio: Ratio of final spread to initial spread.
            Values < 1 indicate convergent training.
        lyapunov: Estimated Lyapunov exponent. Negative values indicate
            stable training, positive values indicate sensitivity.
        behavior: Classification as 'convergent' or 'divergent'.
        spread_evolution: Array of spread values at each checkpoint.
        pca_trajectories: Trajectories projected onto PCA components.
            Shape: (n_checkpoints, n_trajectories, n_components)
        pca_explained_variance: Variance explained by each PCA component.
        effective_dimensionality: Participation ratio measuring degrees of freedom.
        trajectory_metrics: Per-trajectory detailed metrics.
        n_trajectories: Number of trajectories analyzed.
        n_checkpoints: Number of checkpoints recorded.
        n_parameters: Total number of model parameters.
    """

    # Core metrics
    convergence_ratio: float
    lyapunov: float
    behavior: Literal["convergent", "divergent"]

    # Detailed metrics
    spread_evolution: np.ndarray
    pca_trajectories: np.ndarray
    pca_explained_variance: np.ndarray
    effective_dimensionality: float

    # Per-trajectory data
    trajectory_metrics: List[TrajectoryMetrics] = field(default_factory=list)

    # Metadata
    n_trajectories: int = 0
    n_checkpoints: int = 0
    n_parameters: int = 0

    def plot_trajectories(
        self,
        ax: Optional[plt.Axes] = None,
        components: tuple = (0, 1),
        **kwargs: Any,
    ) -> Figure:
        """Plot weight trajectories in PCA space.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.
            components: Which PCA components to plot (default first two).
            **kwargs: Additional arguments passed to plt.plot().

        Returns:
            Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.get_figure()

        colors = plt.cm.viridis(np.linspace(0, 1, self.n_trajectories))

        for i in range(self.n_trajectories):
            trajectory = self.pca_trajectories[:, i, :]
            ax.plot(
                trajectory[:, components[0]],
                trajectory[:, components[1]],
                color=colors[i],
                alpha=0.7,
                label=f"Trajectory {i+1}",
                **kwargs,
            )
            # Mark start and end
            ax.scatter(
                trajectory[0, components[0]],
                trajectory[0, components[1]],
                color=colors[i],
                marker="o",
                s=100,
                edgecolors="black",
                zorder=5,
            )
            ax.scatter(
                trajectory[-1, components[0]],
                trajectory[-1, components[1]],
                color=colors[i],
                marker="s",
                s=100,
                edgecolors="black",
                zorder=5,
            )

        ax.set_xlabel(
            f"PC{components[0]+1} ({self.pca_explained_variance[components[0]]:.1%} var)"
        )
        ax.set_ylabel(
            f"PC{components[1]+1} ({self.pca_explained_variance[components[1]]:.1%} var)"
        )
        ax.set_title(
            f"Weight Trajectories in PCA Space\n(Convergence: {self.convergence_ratio:.2f}x)"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        return fig

    def plot_convergence(
        self,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> Figure:
        """Plot spread evolution over training.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.
            **kwargs: Additional arguments passed to plt.plot().

        Returns:
            Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        epochs = np.arange(len(self.spread_evolution))
        color = "forestgreen" if self.behavior == "convergent" else "crimson"

        ax.plot(epochs, self.spread_evolution, color=color, linewidth=2, **kwargs)
        ax.axhline(
            y=self.spread_evolution[0],
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Initial spread",
        )

        ax.fill_between(
            epochs,
            self.spread_evolution,
            self.spread_evolution[0],
            alpha=0.2,
            color=color,
        )

        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Trajectory Spread (std in PCA space)")
        ax.set_title(
            f"Spread Evolution ({self.behavior.title()})\n"
            f"Ratio: {self.convergence_ratio:.2f}x, λ: {self.lyapunov:.4f}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_lyapunov(
        self,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> Figure:
        """Plot local Lyapunov exponents over training.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.
            **kwargs: Additional arguments passed to plt.plot().

        Returns:
            Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        # Compute local Lyapunov from spread evolution
        spread = self.spread_evolution
        if len(spread) > 1:
            # Local exponent: log(spread[t+1] / spread[t])
            local_lyap = np.log(spread[1:] / np.maximum(spread[:-1], 1e-10))
            epochs = np.arange(1, len(spread))

            ax.bar(
                epochs,
                local_lyap,
                color=["forestgreen" if l < 0 else "crimson" for l in local_lyap],
                alpha=0.7,
                **kwargs,
            )
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
            ax.axhline(
                y=self.lyapunov,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Mean λ = {self.lyapunov:.4f}",
            )

        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Local Lyapunov Exponent")
        ax.set_title("Lyapunov Exponent Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def save_report(
        self,
        output_dir: str,
        embed_images: bool = True,
        format: Literal["markdown", "html"] = "html",
    ) -> None:
        """Generate and save a complete analysis report.

        Args:
            output_dir: Directory to save report files.
            embed_images: If True, embed images as base64 in the report.
                If False, save as separate PNG files with relative links.
            format: Output format - 'html' (default) or 'markdown'.
        """
        import base64
        import io
        import logging

        import matplotlib.pyplot as plt

        logger = logging.getLogger(__name__)

        output_path = Path(output_dir)

        # Validate output_dir doesn't look like a file path
        if output_path.suffix in (".json", ".html", ".md", ".txt", ".png"):
            raise ValueError(
                f"output_dir '{output_dir}' looks like a file path (has extension "
                f"'{output_path.suffix}'). Please provide a directory path instead."
            )

        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving analysis report to: {output_path}")

        # Generate plots and optionally encode as base64
        image_data = {}
        plot_methods = [
            ("trajectories", self.plot_trajectories),
            ("convergence", self.plot_convergence),
            ("lyapunov", self.plot_lyapunov),
        ]

        for name, plot_fn in plot_methods:
            fig = plot_fn()

            if embed_images:
                # Save to buffer and encode as base64
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                image_data[name] = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
            else:
                # Save as separate file
                fig.savefig(output_path / f"{name}.png", dpi=150, bbox_inches="tight")

            plt.close(fig)

        # Save metrics as JSON
        self.to_json(str(output_path / "metrics.json"))
        logger.debug("Saved metrics.json")

        # Generate and save report
        if format == "html":
            report = self._generate_html_report(image_data if embed_images else None)
            report_file = output_path / "report.html"
        else:
            report = self._generate_markdown_report(
                image_data if embed_images else None
            )
            report_file = output_path / "report.md"

        with open(report_file, "w") as f:
            f.write(report)

        logger.info(
            f"Report saved: {report_file} "
            f"({'images embedded' if embed_images else 'images as separate files'})"
        )

    def _generate_markdown_report(
        self, image_data: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate a markdown summary report.

        Args:
            image_data: Optional dict mapping image names to base64 data.
                If provided, images are embedded as data URIs.
        """
        # Create image references - embedded or file-based
        if image_data:
            traj_img = (
                f"![Trajectories](data:image/png;base64,{image_data['trajectories']})"
            )
            conv_img = (
                f"![Convergence](data:image/png;base64,{image_data['convergence']})"
            )
            lyap_img = f"![Lyapunov](data:image/png;base64,{image_data['lyapunov']})"
        else:
            traj_img = "![Trajectories](trajectories.png)"
            conv_img = "![Convergence](convergence.png)"
            lyap_img = "![Lyapunov](lyapunov.png)"

        return f"""# Stability Analysis Report

## Summary

| Metric | Value | Interpretation |
|:-------|------:|:---------------|
| Convergence Ratio | {self.convergence_ratio:.3f}x | {"Convergent - trajectories merge" if self.convergence_ratio < 1 else "Divergent - trajectories separate"} |
| Lyapunov Exponent | {self.lyapunov:.4f} | {"Stable training" if self.lyapunov < 0 else "Sensitive to initialization"} |
| Behavior | {self.behavior.title()} | {"Reproducible results expected" if self.behavior == "convergent" else "Results may vary between runs"} |
| Effective Dimensionality | {self.effective_dimensionality:.1f} | Degrees of freedom in weight dynamics |

## Configuration

- Number of trajectories: {self.n_trajectories}
- Number of checkpoints: {self.n_checkpoints}
- Total parameters: {self.n_parameters:,}

## Visualizations

{traj_img}
*Weight trajectories in PCA space. Circles = start, Squares = end.*

{conv_img}
*Evolution of trajectory spread during training.*

{lyap_img}
*Local Lyapunov exponents at each checkpoint.*

## Interpretation

{"### Convergent Training" if self.behavior == "convergent" else "### Divergent Training"}

{"Training is **stable**: different initializations converge to similar solutions. This architecture is suitable for production deployment where reproducibility is important." if self.behavior == "convergent" else "Training is **sensitive**: small initialization differences lead to distinct solutions. Consider using ensemble methods to leverage this diversity, or investigate if hyperparameters need tuning."}

---
*Generated by deep-lyapunov*
"""

    def _generate_html_report(self, image_data: Optional[Dict[str, str]] = None) -> str:
        """Generate an HTML report with embedded images.

        Args:
            image_data: Optional dict mapping image names to base64 data.
                If provided, images are embedded as data URIs.
        """
        # Create image tags - embedded or file-based
        if image_data:
            traj_img = f'<img src="data:image/png;base64,{image_data["trajectories"]}" alt="Trajectories" style="max-width:100%;">'
            conv_img = f'<img src="data:image/png;base64,{image_data["convergence"]}" alt="Convergence" style="max-width:100%;">'
            lyap_img = f'<img src="data:image/png;base64,{image_data["lyapunov"]}" alt="Lyapunov" style="max-width:100%;">'
        else:
            traj_img = '<img src="trajectories.png" alt="Trajectories" style="max-width:100%;">'
            conv_img = (
                '<img src="convergence.png" alt="Convergence" style="max-width:100%;">'
            )
            lyap_img = '<img src="lyapunov.png" alt="Lyapunov" style="max-width:100%;">'

        behavior_color = "#28a745" if self.behavior == "convergent" else "#dc3545"
        behavior_label = "Convergent" if self.behavior == "convergent" else "Divergent"

        interpretation = (
            "Training is <strong>stable</strong>: different initializations converge to similar solutions. "
            "This architecture is suitable for production deployment where reproducibility is important."
            if self.behavior == "convergent"
            else "Training is <strong>sensitive</strong>: small initialization differences lead to distinct solutions. "
            "Consider using ensemble methods to leverage this diversity, or investigate if hyperparameters need tuning."
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stability Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric-value {{ font-weight: bold; font-family: monospace; }}
        .behavior-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            background-color: {behavior_color};
        }}
        .config-list {{ list-style: none; padding: 0; }}
        .config-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .config-list li:last-child {{ border-bottom: none; }}
        .viz-container {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .viz-container img {{ max-width: 100%; height: auto; }}
        .viz-caption {{
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }}
        .interpretation {{
            background: white;
            padding: 20px;
            border-left: 4px solid {behavior_color};
            margin: 20px 0;
        }}
        footer {{
            text-align: center;
            color: #888;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>Stability Analysis Report</h1>

    <h2>Summary</h2>
    <p>Training behavior: <span class="behavior-badge">{behavior_label}</span></p>

    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Interpretation</th>
        </tr>
        <tr>
            <td>Convergence Ratio</td>
            <td class="metric-value">{self.convergence_ratio:.3f}x</td>
            <td>{"Trajectories merge over time" if self.convergence_ratio < 1 else "Trajectories separate over time"}</td>
        </tr>
        <tr>
            <td>Lyapunov Exponent</td>
            <td class="metric-value">{self.lyapunov:.4f}</td>
            <td>{"Stable training (λ < 0)" if self.lyapunov < 0 else "Sensitive to initialization (λ > 0)"}</td>
        </tr>
        <tr>
            <td>Effective Dimensionality</td>
            <td class="metric-value">{self.effective_dimensionality:.1f}</td>
            <td>Degrees of freedom in weight dynamics</td>
        </tr>
    </table>

    <h2>Configuration</h2>
    <ul class="config-list">
        <li><strong>Trajectories analyzed:</strong> {self.n_trajectories}</li>
        <li><strong>Checkpoints recorded:</strong> {self.n_checkpoints}</li>
        <li><strong>Total parameters:</strong> {self.n_parameters:,}</li>
    </ul>

    <h2>Visualizations</h2>

    <div class="viz-container">
        {traj_img}
        <p class="viz-caption">Weight trajectories in PCA space. Circles = start, Squares = end.</p>
    </div>

    <div class="viz-container">
        {conv_img}
        <p class="viz-caption">Evolution of trajectory spread during training.</p>
    </div>

    <div class="viz-container">
        {lyap_img}
        <p class="viz-caption">Local Lyapunov exponents at each checkpoint.</p>
    </div>

    <h2>Interpretation</h2>
    <div class="interpretation">
        <p>{interpretation}</p>
    </div>

    <footer>
        Generated by <a href="https://github.com/aiexplorations/deep-lyapunov">deep-lyapunov</a>
    </footer>
</body>
</html>
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary.

        Returns:
            Dictionary representation of results.
        """
        return {
            "convergence_ratio": float(self.convergence_ratio),
            "lyapunov": float(self.lyapunov),
            "behavior": self.behavior,
            "effective_dimensionality": float(self.effective_dimensionality),
            "n_trajectories": self.n_trajectories,
            "n_checkpoints": self.n_checkpoints,
            "n_parameters": self.n_parameters,
            "spread_evolution": self.spread_evolution.tolist(),
            "pca_explained_variance": self.pca_explained_variance.tolist(),
            "trajectory_metrics": [asdict(tm) for tm in self.trajectory_metrics],
        }

    def to_json(self, path: str) -> None:
        """Save results to JSON file.

        Args:
            path: Path to save JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnalysisResults:
        """Create AnalysisResults from dictionary.

        Args:
            data: Dictionary with result data.

        Returns:
            AnalysisResults instance.
        """
        trajectory_metrics = [
            TrajectoryMetrics(**tm) for tm in data.get("trajectory_metrics", [])
        ]
        return cls(
            convergence_ratio=data["convergence_ratio"],
            lyapunov=data["lyapunov"],
            behavior=data["behavior"],
            spread_evolution=np.array(data["spread_evolution"]),
            pca_trajectories=np.array(data.get("pca_trajectories", [])),
            pca_explained_variance=np.array(data["pca_explained_variance"]),
            effective_dimensionality=data["effective_dimensionality"],
            trajectory_metrics=trajectory_metrics,
            n_trajectories=data.get("n_trajectories", 0),
            n_checkpoints=data.get("n_checkpoints", 0),
            n_parameters=data.get("n_parameters", 0),
        )
