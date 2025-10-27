import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from types import SimpleNamespace

# python
import torch.nn as nn
from utils import train  # absolute import of the function under test

def make_config(batch_size=4, accumulation_steps=1):
    return SimpleNamespace(batch_size=batch_size, accumulation_steps=accumulation_steps, grad_clip=1.0)

def test_train_runs_with_tuple_batches():
    torch.manual_seed(0)
    N = 12
    input_dim = 8
    num_classes = 3
    inputs = torch.randn(N, input_dim)
    labels = torch.randint(0, num_classes, (N,), dtype=torch.long)

    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    class SimpleModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.linear = nn.Linear(input_dim, num_classes)
        def forward(self, x):
            return self.linear(x)

    model = SimpleModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    config = make_config(batch_size=4)
    result = train(config, loader, model, None, criterion, optimizer, device='cpu')

    assert isinstance(result, tuple) and len(result) == 5
    loss, accuracy, precision, recall, f1 = result
    assert isinstance(loss, float) and loss >= 0.0
    for metric in (accuracy, precision, recall, f1):
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0

def test_train_runs_with_dict_batches():
    torch.manual_seed(1)
    N = 10
    input_dim = 6
    num_classes = 4
    inputs = torch.randn(N, input_dim)
    labels = torch.randint(0, num_classes, (N,), dtype=torch.long)

    class DictDataset(Dataset):
        def __len__(self):
            return N
        def __getitem__(self, idx):
            return {"inputs": inputs[idx], "labels": labels[idx]}

    loader = DataLoader(DictDataset(), batch_size=3, shuffle=False)

    class KwModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.linear = nn.Linear(input_dim, num_classes)
        def forward(self, inputs):
            return self.linear(inputs)

    model = KwModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    config = make_config(batch_size=3)
    result = train(config, loader, model, None, criterion, optimizer, device='cpu')

    assert isinstance(result, tuple) and len(result) == 5
    loss, accuracy, precision, recall, f1 = result
    assert isinstance(loss, float) and loss >= 0.0
    for metric in (accuracy, precision, recall, f1):
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0