from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import datasets, transforms
from omegaconf import DictConfig, OmegaConf
import os
import hydra
import torch
import mlflow

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment('LeNet test')
    root = hydra.utils.get_original_cwd()
    model = TestModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root=os.path.join(root, cfg.dataset.path),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root=os.path.join(root, cfg.dataset.path),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    
    train_dataloader = DataLoader(training_data, batch_size=cfg['training']['batch_size'], shuffle=cfg['training']['shuffle'])
    test_dataloader = DataLoader(test_data, batch_size=cfg['training']['batch_size'], shuffle=cfg['training']['shuffle'])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['optimizer']['lr'])

    mlflow.start_run()

    # Train the model.
    for epoch in range(cfg['training']['epochs']):
        # train(model, train_dataloader, loss_fn, optimizer, device)
        print(f'Epoch {epoch}')
        # test(model, test_dataloader, loss_fn,  device)
        model.train()
        size = len(train_dataloader.dataset)
        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                current = batch_idx * len(X)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
            
        step = epoch
        mlflow.log_metric('loss', loss.item(), step=step)

        model.eval()
        num_batches = len(test_dataloader)
        size = len(test_dataloader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        step = epoch
        mlflow.log_metric('test_loss', test_loss, step=step)
        mlflow.log_metric('test_accuracy', correct, step=step)
    
    mlflow.end_run()


class TestModel(nn.Module):
    def __init__(self) -> None:
        super(TestModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = self.flatten(x)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


if __name__ == '__main__':
    main()