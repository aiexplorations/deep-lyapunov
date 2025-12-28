# Lyapunov Exponents

The mathematical foundation of stability analysis.

## Background

Lyapunov exponents were introduced by Aleksandr Lyapunov in the late 19th century to study the stability of differential equations. They quantify the rate at which nearby trajectories in a dynamical system diverge or converge.

## Definition

For a continuous dynamical system, the **maximal Lyapunov exponent** is defined as:

$$\lambda = \lim_{t \to \infty} \lim_{\delta(0) \to 0} \frac{1}{t} \ln \frac{|\delta(t)|}{|\delta(0)|}$$

where:

- $\delta(0)$ is the initial separation between trajectories
- $\delta(t)$ is the separation at time $t$
- $\lambda$ is the Lyapunov exponent

## Interpretation

| Exponent | System Behavior |
|:---------|:----------------|
| $\lambda < 0$ | **Stable**: Trajectories converge exponentially |
| $\lambda = 0$ | **Neutral**: Trajectories maintain constant separation |
| $\lambda > 0$ | **Chaotic**: Trajectories diverge exponentially |

## For Neural Networks

In neural network training:

- **State**: Weight vector $\theta \in \mathbb{R}^n$
- **Dynamics**: Gradient descent update rule
- **Time**: Training epochs

The Lyapunov exponent measures how sensitive the final trained model is to small changes in initialization.

### Simplified Formula

For practical computation, we use:

$$\lambda \approx \frac{1}{T} \ln \frac{\text{spread}(T)}{\text{spread}(0)}$$

where:

- $T$ is the number of training epochs
- $\text{spread}(t)$ is the standard deviation of weight vectors across perturbed copies at epoch $t$

## Connection to Reproducibility

- **Negative $\lambda$**: Different initializations converge to similar solutions → High reproducibility
- **Positive $\lambda$**: Different initializations lead to different solutions → Low reproducibility

## Local vs Global

### Global Lyapunov Exponent

Average rate over entire training:

$$\lambda_{\text{global}} = \frac{1}{T} \ln \frac{\text{spread}(T)}{\text{spread}(0)}$$

### Local Lyapunov Exponent

Rate at a specific time:

$$\lambda(t) = \ln \frac{\text{spread}(t+1)}{\text{spread}(t)}$$

Local exponents can vary significantly during training, revealing phases of stability and instability.

## Limitations

1. **Finite-time approximation**: We can only compute over finite training runs
2. **Perturbation scale**: Results depend on the initial perturbation magnitude
3. **Stochastic effects**: Mini-batch noise affects trajectory evolution
4. **Non-ergodicity**: Neural network optimization may not satisfy ergodic assumptions

## Further Reading

- Lyapunov, A. M. (1892). "The General Problem of the Stability of Motion"
- Eckmann, J.-P., & Ruelle, D. (1985). "Ergodic theory of chaos and strange attractors"
- Saxe, A. M., et al. (2014). "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
