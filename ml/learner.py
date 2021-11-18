import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.dataset import Dataset


class Learner:
    def __init__(self, model, optimizer=None, criterion=None, device=None):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.device = device or "cpu"
        self.model = self.model.to(device)

    def get_model(self):
        return self.model

    def run(self, dataset: Dataset, mlflow=None):
        """Run learning

        Epoch (or step) is indexed by 1.
        """
        num_epochs = self.model.hyperparams["epochs"]
        for epoch in tqdm(range(1, num_epochs + 1)):
            train_loss, train_acc = self.run_epoch(dataloader=dataset.train_dataloader)
            if mlflow:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
            # Validation
            if dataset.val_dataloader is not None:
                val_loss, val_acc = self.eval(dataloader=dataset.val_dataloader)
                if mlflow:
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_acc", val_acc, step=epoch)

    def run_epoch(self, dataloader: DataLoader):
        """Run one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0.0
        for i, batch in enumerate(dataloader):
            # Send to device
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)
            # Forwarding
            self.optimizer.zero_grad()
            outs = self.model(X)
            loss = self.criterion(outs, y)
            # Backprop
            loss.backward()
            self.optimizer.step()
            # Metrics
            total_loss += loss.detach().item()
            total_correct += torch.sum(torch.argmax(outs, dim=-1) == y).detach().item()

        total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        return total_loss, total_acc

    def eval(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0.0
        for i, batch in enumerate(dataloader):
            # Send to device
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)
            # Eval
            outs = self.model(X)
            loss = self.criterion(outs, y)
            # Metrics
            total_loss += loss.detach().item()
            total_correct += torch.sum(torch.argmax(outs, dim=-1) == y).detach().item()

        total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        return total_loss, total_acc
