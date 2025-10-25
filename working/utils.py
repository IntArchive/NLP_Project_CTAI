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

# def train(config, train_loader, model, decoder, criterion, optimizer, device, accumulation_steps=4):
#     model.train()
#     losses = AverageMeter()
#     acc_meter = AverageMeter()
#     start_time = time.time()

#     optimizer.zero_grad()  # Ensure gradients are cleared at the start

#     for batch_idx, batch in enumerate(train_loader):

#         # loader expected to return (rating, text)
#         if isinstance(batch, (list, tuple)) and len(batch) == 2:
#             labels, texts = batch
#         else:
#             # fallback: dict-like batch
#             labels = batch.get('rating') if hasattr(batch, 'get') else None
#             texts = batch.get('review') if hasattr(batch, 'get') else None

#         # convert labels to tensor and move to device
#         if not torch.is_tensor(labels):
#             labels = torch.tensor(labels, dtype=torch.long)
#         labels = labels.to(device)

#         # convert ratings to 0-based if needed
#         if labels.min() >= 1:
#             labels = labels - 1

#         # ensure texts is a list[str]
#         if isinstance(texts, torch.Tensor):
#             try:
#                 texts = texts.tolist()
#             except Exception:
#                 texts = [str(t) for t in texts]
#         elif not isinstance(texts, (list, tuple)):
#             texts = [str(texts)]

#         # Forward pass
#         outputs = model(texts)
#         loss = criterion(outputs, labels) / accumulation_steps  # Scale loss for accumulation
#         loss = loss.mean()
#         batch_size = labels.size(0)
#         losses.update(loss.item() * accumulation_steps, batch_size)  # Track loss

#         # Backward pass
#         loss.backward()

#         # Perform optimizer step after accumulating gradients
#         if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
#             optimizer.step()
#             optimizer.zero_grad()  # Clear gradients after the step

#         # Calculate accuracy
#         preds = outputs.argmax(dim=1)
#         correct = (preds == labels).sum().item()
#         acc = correct / batch_size
#         acc_meter.update(acc, batch_size)

#     if device == 'cuda':
#         torch.cuda.empty_cache()
#     gc.collect()

#     elapsed = time.time() - start_time
#     print(f"Train Loss: {losses.avg:.4f}  Acc: {acc_meter.avg:.4f}  Time: {elapsed:.1f}s")

#     return losses.avg

def train(config,
          loader,
          model,
          decoder,
          criterion,
          optimizer,
          device):
    

    

    # Ensure criterion uses CrossEntropyLoss if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # choose which module to train (decoder or full model)
    train_module = decoder if decoder is not None else model
    train_module.train()

    model.to(device)
    train_module.to(device)

    # mixed precision scaler if using CUDA
    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    losses = AverageMeter()
    acc_meter = AverageMeter()

    start_time = time.time()
    with torch.enable_grad():
        for _, batch in tqdm(enumerate(loader), total=len(loader)):
            # loader expected to return (rating, text)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                labels, texts = batch
            else:
                # fallback: dict-like batch
                labels = batch.get('rating') if hasattr(batch, 'get') else None
                texts = batch.get('review') if hasattr(batch, 'get') else None

            # convert labels to tensor and move to device
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(device)

            # convert ratings to 0-based if needed
            if labels.min() >= 1:
                labels = labels - 1

            # ensure texts is a list[str]
            if isinstance(texts, torch.Tensor):
                try:
                    texts = texts.tolist()
                except Exception:
                    texts = [str(t) for t in texts]
            elif not isinstance(texts, (list, tuple)):
                texts = [str(texts)]

            accumulation_steps = 4
            optimizer.zero_grad()
            try:
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    outputs = train_module(texts)
                    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    # ensure logits on correct device
                    logits = logits.to(device)
                    loss = criterion(logits, labels)
                    loss = loss.mean()

                
                if (_ + 1) % accumulation_steps == 0:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(train_module.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
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

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            acc = correct / batch_size
            acc_meter.update(acc, batch_size)

    if use_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - start_time
    print(f"Train Loss: {losses.avg:.4f}  Acc: {acc_meter.avg:.4f}  Time: {elapsed:.1f}s")

    return losses.avg

    
    
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
