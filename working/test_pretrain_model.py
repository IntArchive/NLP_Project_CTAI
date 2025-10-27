import torch
from types import SimpleNamespace
from unittest.mock import Mock
import pretrain_model

# python
import torch.nn as nn
from pretrain_model import main  # absolute import of function under test

def test_main_calls_train_and_validate():
    # Prepare config
    config = SimpleNamespace(batch_size=2, num_workers=0, learning_rate=0.01, num_epochs=3, device='cpu')

    # Dummy dataset (minimal)
    class DummyDataset:
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            # return (input, label) - not used because train/validate are mocked
            return torch.randn(3), torch.tensor(0, dtype=torch.long)

    # Dummy model with a parameter so optimizer can be constructed
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))
        def forward(self, x):
            return torch.zeros((x.shape[0], 5)) if isinstance(x, torch.Tensor) else torch.zeros((1,5))

    # Mocks for train/validate
    train_mock = Mock(return_value=(0.1, 0.2, 0.3, 0.4, 0.5))
    validate_mock = Mock(return_value=(0.2, 0.3, 0.4, 0.5, 0.6))

    # Patch module-level dependencies
    pretrain_model.AmazonDataset = lambda cfg, mode='train': DummyDataset()
    pretrain_model.MyModel = DummyModel
    pretrain_model.train = train_mock
    pretrain_model.validate = validate_mock
    pretrain_model.OmegaConf.load = lambda path: config

    # Run main
    main()

    # Assertions: train and validate called once per epoch
    assert train_mock.call_count == config.num_epochs, f"train called {train_mock.call_count} times, expected {config.num_epochs}"
    assert validate_mock.call_count == config.num_epochs, f"validate called {validate_mock.call_count} times, expected {config.num_epochs}"

def test_main_optimizer_filters_requires_grad():
    # Prepare config with single epoch
    config = SimpleNamespace(batch_size=2, num_workers=0, learning_rate=0.01, num_epochs=1, device='cpu')

    # Dummy dataset
    class DummyDataset:
        def __len__(self): return 2
        def __getitem__(self, idx): return torch.randn(3), torch.tensor(0, dtype=torch.long)

    # Dummy model with two parameters: one requires_grad True, one False
    class ModelWithMixedParams(nn.Module):
        def __init__(self):
            super().__init__()
            self.p_true = nn.Parameter(torch.randn(1), requires_grad=True)
            self.p_false = nn.Parameter(torch.randn(1), requires_grad=False)
        def forward(self, x):
            return torch.zeros((x.shape[0], 2)) if isinstance(x, torch.Tensor) else torch.zeros((1,2))

    called = {"ok": False}

    # fake_train inspects optimizer to ensure only the requires_grad=True param was passed
    def fake_train(cfg, loader, model, decoder, criterion, optimizer, device):
        # gather all params passed to optimizer
        opt_params = []
        for g in optimizer.param_groups:
            opt_params.extend(g.get('params', []))
        # Should contain model.p_true but not model.p_false
        has_true = any(torch.equal(p, model.p_true) for p in opt_params)
        has_false = any(torch.equal(p, model.p_false) for p in opt_params)
        assert has_true, "Optimizer should include parameters with requires_grad=True"
        assert not has_false, "Optimizer should not include parameters with requires_grad=False"
        called["ok"] = True
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    # Patch module-level dependencies
    pretrain_model.AmazonDataset = lambda cfg, mode='train': DummyDataset()
    pretrain_model.MyModel = ModelWithMixedParams
    pretrain_model.train = fake_train
    pretrain_model.validate = Mock(return_value=(0.0,0.0,0.0,0.0,0.0))
    pretrain_model.OmegaConf.load = lambda path: config

    # Run main
    main()

    assert called["ok"], "fake_train assertion did not run or failed"

# If running directly, run tests
if __name__ == "__main__":
    test_main_calls_train_and_validate()
    test_main_optimizer_filters_requires_grad()
    print("All tests passed.")