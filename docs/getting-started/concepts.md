# Key Concepts

## Training Stability

Neural network training is a dynamical system: the weights evolve over time according to the optimization algorithm. **Training stability** refers to how this system responds to small perturbations.

### Stable vs Unstable Training

**Stable Training:**

- Small changes in initialization lead to similar final models
- Training is reproducible
- Good for deployment where consistency matters

**Unstable Training:**

- Small changes amplify into large differences
- Different runs produce very different models
- Can be beneficial for ensemble diversity

## Lyapunov Exponents

Lyapunov exponents come from dynamical systems theory. They measure the **rate at which nearby trajectories diverge or converge**.

### Definition

For a dynamical system, the Lyapunov exponent λ is defined as:

$$\lambda = \lim_{t \to \infty} \frac{1}{t} \log \frac{d(t)}{d(0)}$$

where $d(t)$ is the distance between trajectories at time $t$.

### Interpretation

| Lyapunov Exponent | Behavior | Meaning |
|:------------------|:---------|:--------|
| λ < 0 | Convergent | Trajectories merge together |
| λ ≈ 0 | Neutral | Trajectories stay at constant distance |
| λ > 0 | Divergent | Trajectories spread apart exponentially |

## Perturbation-Based Analysis

deep-lyapunov uses **perturbation-based trajectory analysis**:

1. **Start** with a base model initialization
2. **Create** N copies with small Gaussian perturbations (scale ε)
3. **Train** all copies independently on the same data
4. **Track** how the weight trajectories evolve
5. **Compute** stability metrics from trajectory spread

```
Initial State         After Training

   ○ ○                   ○
  ○ ● ○    →→→→→      ○ ● ○    (Convergent)
   ○ ○                   ○

   ○ ○                 ○   ○
  ○ ● ○    →→→→→     ○  ●  ○   (Divergent)
   ○ ○               ○     ○
```

## Key Metrics

### Convergence Ratio

The ratio of final spread to initial spread:

$$\text{ratio} = \frac{\text{std}(\text{final weights})}{\text{std}(\text{initial weights})}$$

- **ratio < 1**: Convergent (stable)
- **ratio > 1**: Divergent (sensitive)

### Effective Dimensionality

Measures how many dimensions are "active" in the weight dynamics. Computed using the **participation ratio** of PCA eigenvalues:

$$\text{PR} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

- **Low PR**: Dynamics confined to few dimensions
- **High PR**: Complex, high-dimensional dynamics

## PCA Projection

Since neural networks have millions of parameters, we project trajectories to a lower-dimensional PCA space for analysis and visualization.

The first few PCA components capture the dominant directions of variation in the weight space, allowing us to visualize and compare trajectories effectively.

## When to Use deep-lyapunov

**Good use cases:**

- Comparing architecture stability
- Debugging training instabilities
- Understanding hyperparameter sensitivity
- Validating reproducibility
- Research on training dynamics

**Consider alternatives if:**

- You only need simple loss curves
- Training is already known to be highly stable
- Computational budget is very limited
