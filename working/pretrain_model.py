from omegaconf import OmegaConf

import pandas as pd
import numpy as np

from nltk.cluster import KMeansClusterer
from sentence_transformers import SentenceTransformer

from utils import AmazonDataset, AmazonPadder, MyModel, train, validate
import torch.nn as nn
import torch


def main():
    config = OmegaConf.load('config.yml')

    train_dataset = AmazonDataset(config, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers)

    val_dataset = AmazonDataset(config, mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             num_workers=config.num_workers)

    
    model = MyModel().to(config.device)
    
    model.train()
    params = model.parameters()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                 lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val_roc = 0
    for i in range(config.num_epochs):
        print(f'Starting epoch {i + 1}')
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(config, train_loader, model, None, criterion, optimizer, config.device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(config, val_loader, model, None, criterion, config.device)
        print(f'Metrics after epoch {i + 1}:\n'
              f'\tTrain loss: {round(train_loss, 3)}\n'
              f'\tTrain accuracy: {round(train_accuracy, 3)}\n'
              f'\tTrain precision: {round(train_precision, 3)}\n'
              f'\tTrain recall: {round(train_recall, 3)}\n'
              f'\tTrain F1: {round(train_f1, 3)}\n'
              f'\tValidation loss: {round(val_loss, 3)}\n'
              f'\tValidation accuracy: {round(val_accuracy, 3)}\n'
              f'\tValidation precision: {round(val_precision, 3)}\n'
              f'\tValidation recall: {round(val_recall, 3)}\n'
              f'\tValidation F1: {round(val_f1, 3)}')
            #   f'\tValidation which: {round(val_roc, 3)}')

        # if val_roc > best_val_roc:
        #     print('New best ROC-AUC, saving model')
        #     best_val_roc = val_roc
        #     if config.tune_8k:
        #         model._model_8k.decoder.load_state_dict(decoder.state_dict())
        #     else:
        #         model._model.decoder.load_state_dict(decoder.state_dict())
        #     torch.jit.save(model, config.model_save_path)
    print('Done')
        
if __name__ == "__main__":
    main()