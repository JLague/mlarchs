from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import torch.nn.functional as F
import torch.nn as nn
import torch
import hydra
import wandb
import models

WANDB_USER = 'jlague'
WANDB_PROJECT = 'mlarchs'


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # WandB setup
    display_name = f'{cfg.training.model}-{datetime.now().strftime("%d-%m-%y_%H-%M-%S")}'
    wandb.init(project=WANDB_PROJECT, entity=WANDB_USER, config=cfg, group=cfg.training.model, name=display_name)

    # Data setup
    project_root = hydra.utils.get_original_cwd()
    train_loader, val_loader, test_loader = load_data(cfg.dataset, project_root)

    # Model setup
    model_class = getattr(models, cfg.training.model)
    model = model_class()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    wandb.watch(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    train(model, train_loader, val_loader, cfg.training.epochs, device, optimizer, loss_fn)
    test(model, test_loader, device, loss_fn)


def load_data(cfg, project_root) -> tuple[DataLoader, DataLoader, DataLoader]:
    match cfg.name:
        case "FashionMNIST":
            dataset = datasets.FashionMNIST
        case "CIFAR10":
            dataset = datasets.CIFAR10

    data_root = f"{project_root}/data"
    generator = torch.Generator().manual_seed(cfg.seed)

    train_data = dataset(
        data_root, train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = dataset(
        data_root, train=False, download=True, transform=transforms.ToTensor()
    )

    train_size = int(cfg.split * len(train_data))
    val_size = len(train_data) - train_size

    train_data, val_data = random_split(train_data, [train_size, val_size])

    loader_args = dict(
        generator=generator,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )

    train_loader = DataLoader(train_data, **loader_args)
    val_loader = DataLoader(val_data, **loader_args)
    test_loader = DataLoader(test_data, **loader_args)

    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, epochs, device, optimizer, loss_fn):
    train_size = len(train_loader)
    val_size = len(val_loader)
    val_data_size = len(val_loader.dataset)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}:')
        train_loss = 0
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'\t Batch {batch_idx}/{train_size}: {train_loss/(batch_idx+1)}')
        
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                loss = loss_fn(pred, y)
                val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
                val_loss += loss.item()

        # Log losses    
        train_loss /= train_size
        val_loss /= val_size
        val_acc /= val_data_size

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch})

        print(f'\t Training loss: {train_loss}')
        print(f'\t Validation loss: {val_loss}')
        print(f'\t Validation accuracy: {val_acc*100}%')

def test(model, dataloader, device, loss_fn):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader.dataset)

    wandb.log({'test_loss': test_loss, 'test_acc': test_acc})

    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_acc*100}%')

if __name__ == '__main__':
    main()