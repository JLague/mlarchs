import math

import torch
import torch.nn as nn
import hydra
import wandb
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import models

WANDB_USER = 'jlague'
WANDB_PROJECT = 'mlarchs'


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # WandB setup
    run_name = f'{cfg.model.display_name}-{datetime.now().strftime("%d-%m-%y_%H-%M-%S")}'
    wandb.init(project=WANDB_PROJECT, entity=WANDB_USER, config=cfg, group=cfg.model.name, name=run_name)
    
    project_root = hydra.utils.get_original_cwd()
    torch.manual_seed(cfg.dataset.seed)
    train_loader, val_loader, test_loader = load_data(cfg.dataset, cfg.model, project_root)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_class = get_module_class(models, cfg.model.name)
    model = model_class(cfg.model.classes)
    model = nn.DataParallel(model)
    model.to(device)
    wandb.watch(model)

    loss_fn = nn.CrossEntropyLoss()

    optim_class = get_module_class(torch.optim, cfg.model.optim)
    optimizer = optim_class(model.parameters(), lr=cfg.model.lr)

    train(model, train_loader, val_loader, cfg.train.epochs, device, optimizer, loss_fn)
    test(model, test_loader, device, loss_fn)


def load_data(dataset_cfg, model_cfg, project_root):

    data_root = f"{project_root}/data"
    
    train_transform = transforms.Compose([
        transforms.Resize(model_cfg.size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(model_cfg.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(model_cfg.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset_class = get_module_class(datasets, model_cfg.dataset)
    train_data = dataset_class(
        data_root, train=True, download=True, transform=train_transform
    )
    test_data = dataset_class(
        data_root, train=False, download=True, transform=test_transform
    )

    train_size = int(dataset_cfg.split * len(train_data))
    val_size = len(train_data) - train_size

    train_data, val_data = random_split(train_data, [train_size, val_size])

    loader_args = dict(
        batch_size=model_cfg.batch_size,
        shuffle=dataset_cfg.shuffle,
    )

    train_loader = DataLoader(train_data, **loader_args)
    val_loader = DataLoader(val_data, **loader_args)
    test_loader = DataLoader(test_data, **loader_args)

    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, epochs, device, optimizer, loss_fn):
    best_loss = math.inf
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

        if val_loss < best_loss:
            best_loss = val_loss
            print(f'Saving model for epoch {epoch+1}')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
            }, 'best_model.pth')

def test(model, dataloader, device, loss_fn):
    test_loss = 0
    test_acc = 0
    model.eval()
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

def get_module_class(module, class_name):
    return getattr(module, class_name)


if __name__ == '__main__':
    main()