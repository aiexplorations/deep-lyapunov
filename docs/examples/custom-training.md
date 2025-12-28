# Custom Training Loop Example

How to use manual recording mode with your own training loop.

## Full Code

```python
--8<-- "examples/custom_training.py"
```

## Key Points

### Getting Model Copies

```python
analyzer.start_recording()
models = analyzer.get_models()
```

You get a list of models to train independently.

### Creating Separate Optimizers

```python
optimizers = [
    torch.optim.Adam(m.parameters(), lr=0.01)
    for m in models
]
```

Each model needs its own optimizer.

### Recording Checkpoints

```python
for epoch in range(n_epochs):
    # Train all models for one epoch
    for model, opt in zip(models, optimizers):
        # ... training ...
        pass

    # Record after each epoch
    analyzer.record_checkpoint()
```

Call `record_checkpoint()` whenever you want to save the current state.

## When to Use This Pattern

- Custom training loops (GANs, RL)
- Distributed training
- Non-standard optimizers
- Custom logging requirements
