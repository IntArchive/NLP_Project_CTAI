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

        self.dataframe = pd.read_csv(dataset_path).reset_index(drop=True).iloc[:200,:]
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

def AmazonPadder(batch):
    ratings = [batch[i][0] for i in range(len(batch))]
    reviews = [batch[i][1] for i in range(len(batch))]
    

    reviews = torch.nn.utils.rnn.pad_sequence(
        reviews, batch_first=True, padding_value=0)

    ratings = torch.nn.utils.rnn.pad_sequence(
        ratings, batch_first=True, padding_value=0)

    return ratings, reviews 

class AverageMeter:
        def __init__(self):
            self.reset()
        def reset(self):
            self.val = 0.0
            self.sum = 0.0
            self.count = 0
            self.avg = 0.0
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / max(1, self.count)
from tqdm import tqdm
import gc
import time
def calculate_metrics(TP, FP, FN):
    precisions, recalls, f1s = [], [], []
    total_tp = sum(TP.values())
    total_fp = sum(FP.values())
    total_fn = sum(FN.values())
    
    # Calculate per-class metrics
    for c in sorted(set(list(TP.keys()) + list(FP.keys()) + list(FN.keys()))):
        tp = TP.get(c, 0)
        fp = FP.get(c, 0)
        fn = FN.get(c, 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    # Calculate macro averages
    if len(precisions) == 0:
        precision = recall = f1 = accuracy = 0.0
    else:
        precision = float(sum(precisions) / len(precisions))
        recall = float(sum(recalls) / len(recalls))
        f1 = float(sum(f1s) / len(f1s))
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    return round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)

# python
def train(config, loader, model, decoder, criterion, optimizer, device, accumulation_steps=None):
    model.train()
    if decoder is not None:
        try:
            decoder.train()
        except Exception:
            pass

    if accumulation_steps is None:
        accumulation_steps = getattr(config, "accumulation_steps", 1)

    use_cuda = device.startswith('cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    losses = AverageMeter()
    acc_meter = AverageMeter()

    # dynamic per-class counters
    TP = {}
    FP = {}
    FN = {}

    start_time = time.time()
    optimizer.zero_grad()
    with torch.enable_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Train")):
            # --- unpack batch (support dict and tuple/list) ---
            if isinstance(batch, dict):
                labels = batch.get('labels')
                print(labels)
                labels = torch.tensor(labels, dtype=torch.long)
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                if labels is None:
                    raise ValueError("Batch dict must contain 'labels' key for training.")
                
                labels = labels.to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs)
            else:
                if isinstance(batch, (list, tuple)):
                    *input_parts, labels = batch
                    print(labels)
                    labels = torch.tensor(labels, dtype=torch.long)
                    
                    labels = labels.to(device)
                    if len(input_parts) == 1:
                        inputs = input_parts[0].to(device)
                        logits = model(inputs)
                    elif len(input_parts) > 1:
                        input_parts = [t.to(device) for t in input_parts]
                        logits = model(*input_parts)
                    else:
                        raise ValueError("Couldn't unpack batch inputs for training.")
                else:
                    raise ValueError("Unexpected batch format in train().")

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            with torch.cuda.amp.autocast(enabled=use_cuda):
                loss = criterion(logits, labels) if criterion is not None else torch.tensor(0.0, device=labels.device)
                if loss.ndim != 0:
                    loss = loss.mean()

            # guard non-finite
            if not torch.isfinite(loss):
                print("WARNING: non-finite loss encountered, skipping batch.")
                optimizer.zero_grad()
                if use_cuda:
                    torch.cuda.empty_cache()
                continue

            loss_for_backprop = loss / accumulation_steps

            try:
                scaler.scale(loss_for_backprop).backward()

                if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    # clip grads safely
                    params_with_grad = [p for p in model.parameters() if p.grad is not None]
                    if len(params_with_grad) > 0:
                        torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=getattr(config, "grad_clip", 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("WARNING: OOM encountered, skipping batch.")
                    optimizer.zero_grad()
                    if use_cuda:
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise

            batch_size = labels.size(0)
            losses.update(loss.item(), batch_size)

            # predictions and update accuracy
            if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
                probs = torch.sigmoid(logits.view(-1))
                preds = (probs > 0.5).long()
            else:
                preds = logits.argmax(dim=1)

            correct = (preds.view(-1) == labels.view(-1)).sum().item()
            acc = correct / batch_size
            acc_meter.update(acc, batch_size)

            # update per-class TP/FP/FN
            pred_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            classes = torch.unique(torch.cat([pred_flat, labels_flat], dim=0)).tolist()
            for c in classes:
                c = int(c)
                tp = ((pred_flat == c) & (labels_flat == c)).sum().item()
                fp = ((pred_flat == c) & (labels_flat != c)).sum().item()
                fn = ((pred_flat != c) & (labels_flat == c)).sum().item()
                TP[c] = TP.get(c, 0) + tp
                FP[c] = FP.get(c, 0) + fp
                FN[c] = FN.get(c, 0) + fn

    if use_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    accuracy, precision, recall, f1 = calculate_metrics(TP, FP, FN)
    
    elapsed = time.time() - start_time
    print(f"Train Loss: {losses.avg:.4f}  Acc: {accuracy:.4f}  P: {precision:.4f}  R: {recall:.4f}  F1: {f1:.4f}  Time: {elapsed:.1f}s")

    return losses.avg, accuracy, precision, recall, f1

# python
def validate(config,
             loader,
             model,
             decoder,
             criterion,
             device):

    losses = AverageMeter()
    acc_meter = AverageMeter()
    TP, FP, FN = {}, {}, {}  # Track per-class metrics
    start_time = time.time()
    batch_size = config.batch_size

    model.eval()
    if decoder is not None:
        try:
            decoder.eval()
        except Exception:
            pass

    use_cuda = device.startswith('cuda')
    torch.cuda.empty_cache()

    with torch.no_grad():
        for batch in loader:
            # Unpack batch into inputs and labels (support dict, tuple/list, tensor)
            if isinstance(batch, dict):
                # assume 'labels' key present
                labels = batch.get('labels')
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                if labels is None:
                    raise ValueError("Batch dict must contain 'labels' key for validation.")
                labels = torch.tensor(labels, dtype=torch.long)
                labels = labels.to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs)
            else:
                # tuple/list: last element is labels
                if isinstance(batch, (list, tuple)):
                    *input_parts, labels = batch
                    labels = torch.tensor(labels, dtype=torch.long)
                    labels = labels.to(device)
                    if len(input_parts) == 1:
                        inputs = input_parts[0].to(device)
                        logits = model(inputs)
                    elif len(input_parts) > 1:
                        input_parts = [t.to(device) for t in input_parts]
                        logits = model(*input_parts)
                    else:
                        raise ValueError("Couldn't unpack batch inputs for validation.")
                else:
                    # single tensor batch: assume (inputs, labels) not followed; cannot handle
                    raise ValueError("Unexpected batch format in validate().")

            # handle models that return tuples (e.g., (logits, ...))
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            # compute loss if criterion provided
            if criterion is not None:
                loss = criterion(logits, labels)
                # if per-sample loss, reduce to scalar
                if loss.ndim != 0:
                    loss = loss.mean()
            else:
                loss = torch.tensor(0.0, device=labels.device)

            # guard against non-finite loss
            if not torch.isfinite(loss):
                print("WARNING: non-finite loss encountered in validation, skipping batch.")
                if use_cuda:
                    torch.cuda.empty_cache()
                continue

            # predictions -> handle binary vs multiclass
            if logits.dim() == 1 or logits.size(1) == 1:
                probs = torch.sigmoid(logits.view(-1))
                preds = (probs > 0.5).long()
            else:
                preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            losses.update(loss.item(), batch_size)
            correct = (preds.view(-1) == labels.view(-1)).sum().item()
            acc = correct / batch_size if batch_size > 0 else 0.0
            acc_meter.update(acc, batch_size)

            # Update per-class metrics
            pred_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            classes = torch.unique(torch.cat([pred_flat, labels_flat])).tolist()
            
            for c in classes:
                c = int(c)
                tp = ((pred_flat == c) & (labels_flat == c)).sum().item()
                fp = ((pred_flat == c) & (labels_flat != c)).sum().item()
                fn = ((pred_flat != c) & (labels_flat == c)).sum().item()
                TP[c] = TP.get(c, 0) + tp
                FP[c] = FP.get(c, 0) + fp
                FN[c] = FN.get(c, 0) + fn

    if use_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    # Calculate final metrics
    accuracy, precision, recall, f1 = calculate_metrics(TP, FP, FN)
    
    elapsed = time.time() - start_time
    print(f"Val Loss: {losses.avg:.4f}  Acc: {accuracy:.4f}  P: {precision:.4f}  R: {recall:.4f}  F1: {f1:.4f}  Time: {elapsed:.1f}s")
    return losses.avg, accuracy, precision, recall, f1
    
    
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

    train(config, loader, model, decoder, criterion, optimizer, device)




if __name__ == "__main__":
    main()
