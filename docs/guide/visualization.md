# Visualization

deep-lyapunov provides rich visualizations to understand training dynamics.

## Built-in Plot Methods

### Trajectory Plot

Shows weight trajectories projected onto the first two PCA components:

```python
fig = results.plot_trajectories()
```

![Trajectories](../assets/trajectories.png)

Features:
- Each color represents one trajectory
- Circles mark starting points
- Squares mark ending points
- Axes show explained variance

### Convergence Plot

Shows how trajectory spread evolves during training:

```python
fig = results.plot_convergence()
```

- Green: Convergent (spread decreasing)
- Red: Divergent (spread increasing)
- Dashed line: Initial spread

### Lyapunov Plot

Shows local Lyapunov exponents at each checkpoint:

```python
fig = results.plot_lyapunov()
```

- Green bars: Negative (stable phase)
- Red bars: Positive (unstable phase)
- Blue dashed line: Global average

## Generating Reports

Create a complete report with all visualizations:

```python
results.save_report("stability_report/")
```

This creates:
- `trajectories.png` - Weight trajectories
- `convergence.png` - Spread evolution
- `lyapunov.png` - Lyapunov evolution
- `metrics.json` - Numerical results
- `report.md` - Markdown summary

## Standalone Visualization Functions

For more control, use the visualization module directly:

```python
from deep_lyapunov.visualization import (
    plot_trajectories_2d,
    plot_trajectories_3d,
    plot_spread_evolution,
    plot_lyapunov_evolution,
    plot_convergence_basin,
    create_analysis_dashboard,
)
```

### 2D Trajectory Plot

```python
fig = plot_trajectories_2d(
    results.pca_trajectories,
    results.pca_explained_variance,
    components=(0, 1),  # Which PCA components
    colormap="viridis",
    show_endpoints=True,
)
```

### 3D Trajectory Plot

```python
fig = plot_trajectories_3d(
    results.pca_trajectories,
    results.pca_explained_variance,
    components=(0, 1, 2),
)
```

### Convergence Basin

Shows initial vs final point distributions:

```python
fig = plot_convergence_basin(
    results.pca_trajectories,
    results.pca_explained_variance,
)
```

### Complete Dashboard

All visualizations in one figure:

```python
fig = create_analysis_dashboard(
    pca_trajectories=results.pca_trajectories,
    pca_variance=results.pca_explained_variance,
    spread_evolution=results.spread_evolution,
    lyapunov=results.lyapunov,
    convergence_ratio=results.convergence_ratio,
    behavior=results.behavior,
    save_path="dashboard.png",
)
```

## Custom Plotting

Use matplotlib with your own styling:

```python
import matplotlib.pyplot as plt

# Create custom figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Use results data directly
results.plot_trajectories(ax=axes[0])
results.plot_convergence(ax=axes[1])

plt.suptitle("My Custom Analysis", fontsize=14)
plt.tight_layout()
plt.savefig("custom_analysis.png", dpi=150)
```

## Animation (Future Feature)

Coming in v0.2: animated trajectory evolution.
