"""Tests for StabilityAnalyzer."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deep_lyapunov import AnalyzerConfig, StabilityAnalyzer


class TestStabilityAnalyzerInit:
    """Tests for StabilityAnalyzer initialization."""

    def test_basic_init(self, simple_model):
        """Test basic analyzer initialization."""
        analyzer = StabilityAnalyzer(simple_model)

        assert analyzer.model is simple_model
        assert analyzer.config.perturbation_scale == 0.01
        assert analyzer.config.n_trajectories == 5

    def test_custom_params(self, simple_model):
        """Test initialization with custom parameters."""
        analyzer = StabilityAnalyzer(
            simple_model,
            perturbation_scale=0.05,
            n_trajectories=10,
            n_pca_components=20,
            device="cpu",
        )

        assert analyzer.config.perturbation_scale == 0.05
        assert analyzer.config.n_trajectories == 10
        assert analyzer.config.n_pca_components == 20

    def test_init_with_config(self, simple_model):
        """Test initialization with AnalyzerConfig."""
        config = AnalyzerConfig(
            perturbation_scale=0.02,
            n_trajectories=8,
            device="cpu",
        )
        analyzer = StabilityAnalyzer(simple_model, config=config)

        assert analyzer.config.perturbation_scale == 0.02
        assert analyzer.config.n_trajectories == 8

    def test_device_resolution_cpu(self, simple_model):
        """Test device resolution."""
        analyzer = StabilityAnalyzer(simple_model, device="cpu")
        assert analyzer._device == "cpu"

    def test_device_resolution_auto(self, simple_model):
        """Test auto device resolution."""
        analyzer = StabilityAnalyzer(simple_model, device="auto")
        # Should resolve to cpu, cuda, or mps
        assert analyzer._device in ["cpu", "cuda", "mps"]


class TestStabilityAnalyzerManualMode:
    """Tests for manual recording mode."""

    def test_start_recording(self, simple_model):
        """Test starting manual recording."""
        analyzer = StabilityAnalyzer(simple_model, device="cpu", verbose=False)
        analyzer.start_recording()

        assert analyzer._is_recording
        assert len(analyzer._models) == analyzer.config.n_trajectories
        assert len(analyzer._trackers) == analyzer.config.n_trajectories

    def test_record_checkpoint(self, simple_model):
        """Test recording checkpoints."""
        analyzer = StabilityAnalyzer(simple_model, device="cpu", verbose=False)
        analyzer.start_recording()

        initial_checkpoints = analyzer._n_checkpoints
        analyzer.record_checkpoint()

        assert analyzer._n_checkpoints == initial_checkpoints + 1

    def test_record_without_start_raises(self, simple_model):
        """Test that recording without starting raises error."""
        analyzer = StabilityAnalyzer(simple_model, device="cpu", verbose=False)

        with pytest.raises(RuntimeError, match="Not recording"):
            analyzer.record_checkpoint()

    def test_start_recording_twice_raises(self, simple_model):
        """Test that starting recording twice raises error."""
        analyzer = StabilityAnalyzer(simple_model, device="cpu", verbose=False)
        analyzer.start_recording()

        with pytest.raises(RuntimeError, match="Already recording"):
            analyzer.start_recording()

    def test_get_models(self, simple_model):
        """Test getting model copies."""
        analyzer = StabilityAnalyzer(
            simple_model, device="cpu", verbose=False, n_trajectories=3
        )
        analyzer.start_recording()

        models = analyzer.get_models()
        assert len(models) == 3
        assert all(isinstance(m, nn.Module) for m in models)

    def test_get_models_before_start_raises(self, simple_model):
        """Test getting models before starting raises error."""
        analyzer = StabilityAnalyzer(simple_model, device="cpu", verbose=False)

        with pytest.raises(RuntimeError, match="Models not created"):
            analyzer.get_models()

    def test_reset(self, simple_model):
        """Test analyzer reset."""
        analyzer = StabilityAnalyzer(simple_model, device="cpu", verbose=False)
        analyzer.start_recording()
        analyzer.record_checkpoint()
        analyzer.reset()

        assert not analyzer._is_recording
        assert len(analyzer._models) == 0
        assert len(analyzer._trackers) == 0
        assert analyzer._n_checkpoints == 0


class TestStabilityAnalyzerComputeMetrics:
    """Tests for compute_metrics."""

    def test_compute_metrics_basic(self, simple_model):
        """Test basic metrics computation."""
        analyzer = StabilityAnalyzer(
            simple_model,
            device="cpu",
            verbose=False,
            n_trajectories=3,
        )
        analyzer.start_recording()

        # Record several checkpoints with weight modifications
        for _ in range(5):
            # Modify weights slightly
            with torch.no_grad():
                for model in analyzer.get_models():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)
            analyzer.record_checkpoint()

        results = analyzer.compute_metrics()

        assert results.n_trajectories == 3
        assert results.n_checkpoints == 6  # Initial + 5 recorded
        assert results.convergence_ratio > 0
        assert results.behavior in ["convergent", "divergent"]
        assert len(results.spread_evolution) == 6

    def test_compute_metrics_insufficient_data_raises(self, simple_model):
        """Test that insufficient data raises error."""
        analyzer = StabilityAnalyzer(simple_model, device="cpu", verbose=False)
        analyzer.start_recording()
        # Only initial checkpoint, need at least 2

        with pytest.raises(RuntimeError, match="Insufficient trajectory data"):
            analyzer.compute_metrics()


class TestStabilityAnalyzerAnalyze:
    """Tests for automatic analyze mode."""

    @pytest.fixture
    def stable_model(self):
        """Create a stable model for analyze tests."""
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        # Initialize with reasonable values
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        return model

    @pytest.fixture
    def stable_dataloader(self):
        """Create stable training data."""
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(42)
        X = torch.randn(40, 4)
        y = torch.randn(40, 2)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=10, shuffle=False)

    @pytest.fixture
    def simple_train_fn(self):
        """Create a simple training function."""

        def train_fn(model, train_loader, n_epochs, **kwargs):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Lower LR
            criterion = nn.MSELoss()
            losses = []

            for _ in range(n_epochs):
                epoch_loss = 0.0
                for X, y in train_loader:
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(train_loader))

            return {"loss": losses}

        return train_fn

    def test_analyze_basic(self, stable_model, stable_dataloader, simple_train_fn):
        """Test basic analyze workflow."""
        analyzer = StabilityAnalyzer(
            stable_model,
            device="cpu",
            verbose=False,
            n_trajectories=2,
            perturbation_scale=0.001,  # Smaller perturbation
        )

        results = analyzer.analyze(
            train_fn=simple_train_fn,
            train_loader=stable_dataloader,
            n_epochs=3,
        )

        assert results.n_trajectories == 2
        assert results.n_checkpoints == 4  # Initial + 3 epochs
        assert results.convergence_ratio > 0
        assert results.n_parameters > 0

    def test_analyze_with_training_metrics(
        self, stable_model, stable_dataloader, simple_train_fn
    ):
        """Test that training metrics are captured."""
        analyzer = StabilityAnalyzer(
            stable_model,
            device="cpu",
            verbose=False,
            n_trajectories=2,
            perturbation_scale=0.001,
        )

        results = analyzer.analyze(
            train_fn=simple_train_fn,
            train_loader=stable_dataloader,
            n_epochs=3,
        )

        # Check trajectory metrics
        assert len(results.trajectory_metrics) == 2
        for tm in results.trajectory_metrics:
            assert tm.final_loss is not None or tm.path_length >= 0

    def test_analyze_logging_output(self, stable_model, stable_dataloader, caplog):
        """Test logging output during analyze."""
        import logging

        def simple_train_fn(model, train_loader, n_epochs, **kwargs):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            losses = []
            for _ in range(n_epochs):
                epoch_loss = 0.0
                for X, y in train_loader:
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(train_loader))
            return {"loss": losses}

        analyzer = StabilityAnalyzer(
            stable_model,
            device="cpu",
            verbose=True,
            n_trajectories=2,
            perturbation_scale=0.001,
        )

        with caplog.at_level(logging.INFO, logger="deep_lyapunov"):
            results = analyzer.analyze(
                train_fn=simple_train_fn,
                train_loader=stable_dataloader,
                n_epochs=2,
            )

        # Check logging output from analyze()
        log_text = caplog.text
        assert "Starting stability analysis" in log_text
        assert "Training trajectory" in log_text
        assert "Analysis complete" in log_text

    def test_analyze_with_accuracy_metrics(self, stable_model, stable_dataloader):
        """Test analyze with accuracy in training results (lines 271-276)."""

        def train_fn_with_accuracy(model, train_loader, n_epochs, **kwargs):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            losses = []
            accuracies = []
            for _ in range(n_epochs):
                epoch_loss = 0.0
                for X, y in train_loader:
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(train_loader))
                accuracies.append(0.8 + len(losses) * 0.05)  # Fake accuracy
            return {"loss": losses, "accuracy": accuracies}

        analyzer = StabilityAnalyzer(
            stable_model,
            device="cpu",
            verbose=False,
            n_trajectories=2,
            perturbation_scale=0.001,
        )

        results = analyzer.analyze(
            train_fn=train_fn_with_accuracy,
            train_loader=stable_dataloader,
            n_epochs=2,
        )

        # Check that accuracy was captured
        for tm in results.trajectory_metrics:
            assert tm.final_accuracy is not None

    def test_analyze_with_test_acc_metrics(self, stable_model, stable_dataloader):
        """Test analyze with test_acc in training results (lines 278-283)."""

        def train_fn_with_test_acc(model, train_loader, n_epochs, **kwargs):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            losses = []
            test_accs = []
            for _ in range(n_epochs):
                epoch_loss = 0.0
                for X, y in train_loader:
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(train_loader))
                test_accs.append(0.75 + len(losses) * 0.05)  # Fake test_acc
            return {"loss": losses, "test_acc": test_accs}

        analyzer = StabilityAnalyzer(
            stable_model,
            device="cpu",
            verbose=False,
            n_trajectories=2,
            perturbation_scale=0.001,
        )

        results = analyzer.analyze(
            train_fn=train_fn_with_test_acc,
            train_loader=stable_dataloader,
            n_epochs=2,
        )

        # Check that test_acc was captured as accuracy
        for tm in results.trajectory_metrics:
            assert tm.final_accuracy is not None


class TestStabilityAnalyzerHelpers:
    """Tests for helper methods."""

    def test_collect_trajectories(self, simple_model):
        """Test trajectory collection."""
        analyzer = StabilityAnalyzer(
            simple_model,
            device="cpu",
            verbose=False,
            n_trajectories=3,
        )
        analyzer.start_recording()

        # Record checkpoints
        for _ in range(4):
            analyzer.record_checkpoint()

        trajectories = analyzer._collect_trajectories()

        # Shape should be (n_checkpoints, n_trajectories, n_params)
        assert trajectories.shape[0] == 5  # Initial + 4
        assert trajectories.shape[1] == 3
        assert trajectories.shape[2] > 0

    def test_compute_trajectory_metrics(self, simple_model):
        """Test per-trajectory metrics computation."""
        analyzer = StabilityAnalyzer(
            simple_model,
            device="cpu",
            verbose=False,
            n_trajectories=2,
        )
        analyzer.start_recording()

        # Record with modifications
        for _ in range(3):
            with torch.no_grad():
                for model in analyzer.get_models():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)
            analyzer.record_checkpoint()

        training_metrics = [
            {"loss": [0.5, 0.4, 0.3], "accuracy": [0.8, 0.85, 0.9]},
            {"loss": [0.6, 0.45, 0.35], "accuracy": [0.75, 0.82, 0.88]},
        ]

        metrics = analyzer._compute_trajectory_metrics(training_metrics)

        assert len(metrics) == 2
        assert metrics[0].final_loss == 0.3
        assert metrics[0].final_accuracy == 0.9
        assert metrics[1].final_loss == 0.35


class TestStabilityAnalyzerLogging:
    """Tests for logging output."""

    def test_logging_on_start_recording(self, simple_model, caplog):
        """Test that logging produces output on start_recording."""
        import logging

        analyzer = StabilityAnalyzer(simple_model, device="cpu", verbose=True)

        with caplog.at_level(logging.INFO, logger="deep_lyapunov"):
            analyzer.start_recording()

        assert "Recording started" in caplog.text

    def test_logging_on_reset(self, simple_model, caplog):
        """Test logging output during reset."""
        import logging

        analyzer = StabilityAnalyzer(simple_model, device="cpu", verbose=True)
        analyzer.start_recording()
        caplog.clear()

        with caplog.at_level(logging.INFO, logger="deep_lyapunov"):
            analyzer.reset()

        assert "Analyzer reset" in caplog.text

    def test_logging_on_compute_metrics(self, simple_model, caplog):
        """Test logging output during compute_metrics."""
        import logging

        analyzer = StabilityAnalyzer(
            simple_model,
            device="cpu",
            verbose=True,
            n_trajectories=3,
        )
        analyzer.start_recording()
        caplog.clear()

        # Record several checkpoints with weight modifications
        for _ in range(3):
            with torch.no_grad():
                for model in analyzer.get_models():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)
            analyzer.record_checkpoint()

        with caplog.at_level(logging.INFO, logger="deep_lyapunov"):
            results = analyzer.compute_metrics()

        log_text = caplog.text
        assert "Analysis complete" in log_text
        assert "convergence_ratio" in log_text
        assert "lyapunov" in log_text

    def test_debug_logging(self, simple_model, caplog):
        """Test that debug level provides more details."""
        import logging

        analyzer = StabilityAnalyzer(
            simple_model,
            device="cpu",
            n_trajectories=2,
        )

        with caplog.at_level(logging.DEBUG, logger="deep_lyapunov"):
            analyzer.start_recording()

        # Debug should include more detailed info
        log_text = caplog.text
        assert "perturbed model copies" in log_text or "parameters" in log_text


class TestStabilityAnalyzerTrainingMetrics:
    """Tests for training metrics extraction."""

    def test_extract_accuracy_from_training_results(self, simple_model):
        """Test extracting accuracy from training results."""
        analyzer = StabilityAnalyzer(
            simple_model,
            device="cpu",
            verbose=False,
            n_trajectories=2,
        )
        analyzer.start_recording()

        # Record checkpoints
        for _ in range(3):
            with torch.no_grad():
                for model in analyzer.get_models():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)
            analyzer.record_checkpoint()

        # Test with accuracy in training_metrics
        training_metrics = [
            {"loss": [0.5, 0.4], "accuracy": [0.8, 0.85]},
            {"loss": [0.6, 0.45], "accuracy": [0.75, 0.82]},
        ]

        metrics = analyzer._compute_trajectory_metrics(training_metrics)
        assert metrics[0].final_accuracy == 0.85
        assert metrics[1].final_accuracy == 0.82

    def test_extract_no_accuracy_from_training_results(self, simple_model):
        """Test when training metrics have no accuracy."""
        analyzer = StabilityAnalyzer(
            simple_model,
            device="cpu",
            verbose=False,
            n_trajectories=2,
        )
        analyzer.start_recording()

        # Record checkpoints
        for _ in range(3):
            with torch.no_grad():
                for model in analyzer.get_models():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)
            analyzer.record_checkpoint()

        # Test with only loss, no accuracy
        training_metrics = [
            {"loss": [0.5, 0.4]},
            {"loss": [0.6, 0.45]},
        ]

        metrics = analyzer._compute_trajectory_metrics(training_metrics)
        assert metrics[0].final_loss == 0.4
        assert metrics[0].final_accuracy is None
        assert metrics[1].final_loss == 0.45

    def test_extract_empty_loss_list(self, simple_model):
        """Test extracting metrics when loss list is empty."""
        analyzer = StabilityAnalyzer(
            simple_model,
            device="cpu",
            verbose=False,
            n_trajectories=2,
        )
        analyzer.start_recording()

        # Record checkpoints
        for _ in range(3):
            with torch.no_grad():
                for model in analyzer.get_models():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * 0.01)
            analyzer.record_checkpoint()

        # Test with empty lists (should not crash)
        training_metrics = [
            {"loss": [], "accuracy": []},
            {"loss": [0.35], "accuracy": [0.88]},
        ]

        metrics = analyzer._compute_trajectory_metrics(training_metrics)
        # Empty list is falsy, so final_loss should be None
        assert metrics[0].final_loss is None
        assert metrics[1].final_loss == 0.35
        assert metrics[1].final_accuracy == 0.88
