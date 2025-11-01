from omegaconf import OmegaConf
import torch
import torch.nn as nn

from utils import AmazonDataset, collate_text_batch, MyModel, train, validate, EarlyStopping


def main():
    config = OmegaConf.load("config.yml")

    # Defaults nếu thiếu
    if "device" not in config:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    if "batch_size" not in config:
        config.batch_size = 8
    if "num_workers" not in config:
        config.num_workers = 0
    if "learning_rate" not in config:
        config.learning_rate = 2e-4
    if "num_epochs" not in config:
        config.num_epochs = 5
    if "num_classes" not in config:
        config.num_classes = 5
    if "use_amp" not in config:
        config.use_amp = True
    if "accumulation_steps" not in config:
        config.accumulation_steps = 1
    if "patience" not in config:
        config.patience = 3
    if "min_delta" not in config:
        config.min_delta = 0.0
    if "model_save_path" not in config:
        config.model_save_path = "best_model.pt"

    # Datasets/Loaders
    train_dataset = AmazonDataset(config, mode="train")
    val_dataset = AmazonDataset(config, mode="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        collate_fn=collate_text_batch,
        drop_last=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        collate_fn=collate_text_batch,
        drop_last=False,
    )

    # Model + Optim/Loss
    model = MyModel(num_classes=config.num_classes, device=config.device).to(config.device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.learning_rate),
    )
    criterion = nn.CrossEntropyLoss()  # dùng mean cho ổn định

    # Early stopper dựa vào val_loss (mode='min')
    early_stopper = EarlyStopping(patience=config.patience, min_delta=config.min_delta, mode="min")

    # Loop
    for epoch in range(int(config.num_epochs)):
        print(f"Starting epoch {epoch + 1}")
        train_loss = train(config, train_loader, model, None, criterion, optimizer, config.device)

        val_loss, val_metrics, improved, should_stop = validate(
            config,
            val_loader,
            model,
            None,
            criterion,
            config.device,
            early_stopper=early_stopper,
            save_path=config.model_save_path,
        )

        print(
            f"Metrics after epoch {epoch + 1}:\n"
            f"\tTrain loss: {round(float(train_loss), 3)}\n"
            f"\tVal   loss: {round(float(val_loss), 3)} | "
            f"Acc: {val_metrics['accuracy']:.3f}  P:{val_metrics['precision']:.3f} "
            f"R:{val_metrics['recall']:.3f}  F1:{val_metrics['f1']:.3f}\n"
        )

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best val loss: {early_stopper.best:.4f}")
            break

    # (Optional) load best checkpoint để dùng inference/eval sau training
    try:
        model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
        print(f"Loaded best checkpoint from {config.model_save_path}")
    except Exception as e:
        print(f"Skip loading best checkpoint: {e}")

    print("Done")


if __name__ == "__main__":
    main()
