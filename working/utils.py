import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
# Requires transformers>=4.51.0
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class AmazonDataset(Dataset):
    def __init__(self,
                 config,
                 mode='train'):
        super(AmazonDataset,self).__init__()
        assert mode in ['train', 'val']
        dataset_path = config.train_dataset_path if mode == 'train' else config.val_dataset_path

        self.dataframe = pd.read_csv(dataset_path).reset_index(drop=True)
        self.index_dict = self.dataframe.to_dict('index')
        self.mode = mode
        print(f'DATASET SIZE : {len(self.dataframe)}')

    def __getitem__(self, idx):
        idx = None if self.mode == 'train' else idx
        rating, text = self.load_text_sample(idx)

        if self.mode == 'train':
            text = self.add_augs(text)
        return rating, text

    def __len__(self):
        return len(self.index_dict)
    
    def load_text_sample(self, idx):
        if idx is None:
            sample = self.dataframe.sample(n=1).iloc[0]
        else:
            sample = self.index_dict[idx]
        rating = sample['rating']
        text = sample['review']
        return rating, text
    
    def add_augs(self, text):
        # Placeholder for augmentation logic
        return text




class QwenB(nn.Module):
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', device: str = 'cuda'):
        super(QwenB, self).__init__()
        # Model initialization code here
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = 8192
        
        if device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.to(device)

    def forward(self,
                input_texts: list[str]) -> Tensor:
        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings
    
    def last_token_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    # def get_detailed_instruct(task_description: str, query: str) -> str:
    #     return f'Instruct: {task_description}\nQuery:{query}'
    
class MyModel(nn.Module):
    def __init__(self, input_dim=1024, hidden=(512, 256, 128), num_classes=5, p_drop=0.2):
        super().__init__()
        self.emodel = QwenB()
        self.norm = nn.LayerNorm(input_dim)
        layers = []
        dims = [input_dim, *hidden]
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True), nn.Dropout(p_drop)]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], num_classes)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.emodel(x)  # (B,1024)
        x = self.norm(x)
        x = self.mlp(x)
        return self.head(x)  # logits (B,5)

import os, gc, time, math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================
# Dataset
# =========================
class AmazonDataset(Dataset):
    def __init__(self, config, mode: str = "train"):
        super().__init__()
        assert mode in ["train", "val"]
        dataset_path = config.train_dataset_path if mode == "train" else config.val_dataset_path

        self.dataframe = pd.read_csv(dataset_path).reset_index(drop=True)
        self.index_dict = self.dataframe.to_dict("index")
        self.mode = mode
        print(f"DATASET SIZE : {len(self.dataframe)}")

    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, idx):
        rating, text = self.load_text_sample(idx)
        if self.mode == "train":
            text = self.add_augs(text)
        return rating, text

    def load_text_sample(self, idx):
        sample = self.index_dict[idx]
        rating = sample["rating"]
        text = sample["review"]
        return rating, text

    def add_augs(self, text: str) -> str:
        return text


# Collate: trả về (labels: LongTensor, texts: list[str])
def collate_text_batch(batch):
    labels = torch.tensor([int(b[0]) for b in batch], dtype=torch.long)
    texts = [str(b[1]) for b in batch]
    return labels, texts


# =========================
# Qwen Embedding Backbone
# =========================
class QwenB(nn.Module):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cuda", max_length: int = 1024):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(device)

    @torch.no_grad()
    def forward(self, input_texts: list[str]) -> Tensor:
        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        emb = self.last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        return emb.to(dtype=torch.float32)

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            seq_len = attention_mask.sum(dim=1) - 1
            bsz = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), seq_len]


# =========================
# Classifier
# =========================
class MyModel(nn.Module):
    def __init__(self, input_dim: int | None = None, hidden=(512, 256, 128), num_classes=5, p_drop=0.2, device: str = "cuda"):
        super().__init__()
        self.emodel = QwenB(device=device)

        backbone_dim = getattr(getattr(self.emodel, "model", None), "config", None)
        backbone_dim = getattr(backbone_dim, "hidden_size", None)
        self.embed_dim = input_dim or backbone_dim or 1024

        self.norm = nn.LayerNorm(self.embed_dim)
        dims = [self.embed_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True), nn.Dropout(p_drop)]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], num_classes)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, texts: list[str]) -> Tensor:
        x = self.emodel(texts)             # (B, embed_dim)
        x = self.norm(x)
        x = self.mlp(x)
        logits = self.head(x)              # (B, num_classes)
        return logits


# =========================
# AverageMeter
# =========================
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0
    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


# =========================
# EarlyStopping helper
# =========================
class EarlyStopping:
    """
    Theo dõi một đại lượng (mặc định: val_loss) và dừng sớm nếu không cải thiện sau `patience` lần.
    mode='min' dùng cho loss, 'max' dùng cho metric dạng càng lớn càng tốt.
    """
    def __init__(self, patience=3, min_delta=0.0, mode="min"):
        assert mode in ("min", "max")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs = 0
        if mode == "min":
            self._is_better = lambda current, best: current < best - self.min_delta
        else:
            self._is_better = lambda current, best: current > best + self.min_delta

    def step(self, current_value: float):
        improved = False
        if self._is_better(current_value, self.best):
            self.best = current_value
            self.num_bad_epochs = 0
            improved = True
        else:
            self.num_bad_epochs += 1
        should_stop = self.num_bad_epochs >= self.patience
        return improved, should_stop


# =========================
# Train
# =========================
def train(config,
          loader: DataLoader,
          model: nn.Module,
          decoder,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: str):

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    use_amp = (device.startswith("cuda") and torch.cuda.is_available() and getattr(config, "use_amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    accumulation_steps = int(getattr(config, "accumulation_steps", 1))
    losses = AverageMeter()
    acc_meter = AverageMeter()

    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in tqdm(enumerate(loader), total=len(loader)):
        labels, texts = batch

        labels = labels.to(device)
        if labels.numel() > 0 and labels.min() >= 1:
            labels = labels - 1  # 1..5 -> 0..4

        num_classes = getattr(config, "num_classes", 5)
        if (labels < 0).any() or (labels >= num_classes).any():
            uniq = labels.detach().cpu().unique().tolist()
            print(f"[skip batch {step}] invalid labels {uniq} for num_classes={num_classes}")
            optimizer.zero_grad(set_to_none=True)
            continue

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(texts)
            if not torch.isfinite(logits).all():
                lo = torch.nanmin(logits).item() if torch.isnan(logits).any() else logits.min().item()
                hi = torch.nanmax(logits).item() if torch.isnan(logits).any() else logits.max().item()
                print(f"[skip batch {step}] non-finite logits in range [{lo:.2e}, {hi:.2e}]")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = criterion(logits, labels)
            if loss.dim() > 0:
                loss = loss.mean()
            loss_for_backward = loss / max(1, accumulation_steps)

        if not torch.isfinite(loss_for_backward):
            print(f"[skip batch {step}] non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            continue

        torch.cuda.empty_cache() if device.startswith("cuda") else None
        scaler.scale(loss_for_backward).backward()

        do_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(loader))
        if do_step:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bsz = labels.size(0)
        losses.update(loss.detach().item(), bsz)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()
        acc_meter.update(acc, bsz)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - start_time
    print(f"Train Loss: {losses.avg:.4f}  Acc: {acc_meter.avg:.4f}  Time: {elapsed:.1f}s")

    return losses.avg


# =========================
# Validate + Early Stopping
# =========================
def validate(config,
             loader: DataLoader,
             model: nn.Module,
             decoder,                  # để tương thích với chữ ký cũ (không dùng)
             criterion: nn.Module,
             device: str,
             early_stopper: EarlyStopping | None = None,
             save_path: str | None = None):
    """
    Trả về:
      val_loss (float),
      metrics (dict: accuracy/precision/recall/f1),
      improved (bool: có cải thiện theo tiêu chí early_stopper),
      should_stop (bool: nên dừng sớm).
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.eval()

    use_amp = (device.startswith("cuda") and torch.cuda.is_available() and getattr(config, "use_amp", True))
    losses = AverageMeter()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader), total=len(loader)):
            labels, texts = batch
            labels = labels.to(device)
            if labels.numel() > 0 and labels.min() >= 1:
                labels = labels - 1

            num_classes = getattr(config, "num_classes", 5)
            if (labels < 0).any() or (labels >= num_classes).any():
                uniq = labels.detach().cpu().unique().tolist()
                print(f"[val skip batch {step}] invalid labels {uniq}")
                continue

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(texts)
                if not torch.isfinite(logits).all():
                    lo = torch.nanmin(logits).item() if torch.isnan(logits).any() else logits.min().item()
                    hi = torch.nanmax(logits).item() if torch.isnan(logits).any() else logits.max().item()
                    print(f"[val skip batch {step}] non-finite logits in range [{lo:.2e}, {hi:.2e}]")
                    continue

                loss = criterion(logits, labels)
                if loss.dim() > 0:
                    loss = loss.mean()

            bsz = labels.size(0)
            losses.update(loss.detach().item(), bsz)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if len(all_preds) == 0:
        # không có batch hợp lệ
        acc = prec = rec = f1 = 0.0
    else:
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        acc = float((y_pred == y_true).mean())
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

    improved = False
    should_stop = False
    if early_stopper is not None:
        # Early stopping dựa vào val_loss (mode='min')
        improved, should_stop = early_stopper.step(losses.avg)
        if improved and save_path:
            # Lưu checkpoint tốt nhất
            torch.save(model.state_dict(), save_path)

    print(
        f"Validation: loss={losses.avg:.4f}  acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}"
        + ("  (improved ✔)" if improved else "")
        + ("  (early stop ✖)" if should_stop else "")
    )

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    return losses.avg, metrics, improved, should_stop


    
    
def main():
    from omegaconf import OmegaConf
    # Load configuration
    config = OmegaConf.load('config.yml')
    # Load dataset
    dataset = AmazonDataset(config, mode='train')
    texts = [dataset[i][1] for i in range(4)]

    model = QwenB()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)




if __name__ == "__main__":
    main()
