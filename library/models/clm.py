import json
import os
from typing import Dict, List

import torch
import tqdm
from torch import nn
from torch.nn import functional as F

from library.data.dataloaders import get_dataloader
from library.training.callbacks import Callback


class ChemicalLanguageModel(nn.Module):
    def __init__(
        self,
        n_layers: int,
        model_dim: int,
        dropout: float,
        vocab_size: int,
        sequence_length: int,
        learning_rate: float,
        n_max_epochs: int,
        batch_size: int,
        device: str,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.model_dim = model_dim
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.n_max_epochs = n_max_epochs
        self.batch_size = batch_size
        self.device = device

        self.architecture = self.build_architecture()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/model.pt")
        fields = self.__dict__.copy()
        init_arguments = {
            field: value
            for field, value in fields.items()
            if field[0] != "_" and field != "training"
        }
        with open(f"{path}/init_arguments.json", "w") as f:
            json.dump(init_arguments, f, indent=4)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = None):
        with open(f"{path}/init_arguments.json", "r") as f:
            init_arguments = json.load(f)
        if device is not None:
            init_arguments["device"] = device
        else:
            device = init_arguments["device"]
        model = cls(**init_arguments)
        model.load_state_dict(torch.load(f"{path}/model.pt"))
        return model.to(device)

    def get_n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, inputs, hidden_states=None, training: bool = True):
        raise NotImplementedError

    def __compute_loss(
        self, inputs: torch.LongTensor, targets: torch.LongTensor
    ) -> torch.Tensor:
        logits = self.forward(
            inputs, training=True
        )  # (batch_size, sequence_length, vocab_size)
        logits = logits.permute(0, 2, 1)  # (batch_size, vocab_size, sequence_length)
        return F.cross_entropy(
            logits,
            targets.long(),
            ignore_index=0,
            reduction="mean",
        )

    def fit(
        self,
        training_molecules_path: str,
        val_molecules_path: str,
        token2label: Dict[str, int],
        starting_epoch: int = 0,
        callbacks: List[Callback] = None,
        fraction_of_dataset: float = None,
        smiles_enumeration: bool = False,
    ) -> Dict[str, List[float]]:
        if callbacks is None:
            callbacks = list()
        # self.architecture = self.build_architecture()
        self = torch.compile(self)
        self = self.to(self.device)
        self.train()
        train_dataloader = get_dataloader(
            training_molecules_path,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            num_workers=8,
            shuffle=True,
            token2label=token2label,
            fraction_of_dataset=fraction_of_dataset,
            smiles_enumeration=smiles_enumeration,
        )

        val_dataloader = get_dataloader(
            val_molecules_path,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            num_workers=8,
            shuffle=True,
            token2label=token2label,
            fraction_of_dataset=fraction_of_dataset,
            smiles_enumeration=smiles_enumeration,
        )
        print(f"Number of parameters: {self.get_n_parameters() / 1e6:.2f}M")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        history = {"train_loss": list(), "val_loss": list()}
        epoch_train_loss = 0
        for epoch_ix in range(starting_epoch, self.n_max_epochs):
            self.train()
            n_train_samples, epoch_train_loss = 0, 0
            for X_train, y_train in tqdm.tqdm(train_dataloader):
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)
                n_train_samples += X_train.shape[0]
                optimizer.zero_grad()
                batch_train_loss = self.__compute_loss(X_train, y_train)
                batch_train_loss.backward()
                optimizer.step()
                epoch_train_loss += batch_train_loss.item() * X_train.shape[0]

            epoch_train_loss = epoch_train_loss / n_train_samples
            history["train_loss"].append(epoch_train_loss)

            self.eval()
            n_val_samples, epoch_val_loss = 0, 0
            for X_val, y_val in val_dataloader:
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                n_val_samples += X_val.shape[0]
                batch_val_loss = self.__compute_loss(X_val, y_val)
                epoch_val_loss += batch_val_loss.item() * X_val.shape[0]

            epoch_val_loss = epoch_val_loss / n_val_samples
            history["val_loss"].append(epoch_val_loss)

            # Callbacks
            print(
                f"Epoch:{epoch_ix}\tLoss: {epoch_train_loss}, Val Loss: {epoch_val_loss}"
            )
            stop_training = False
            for callback in callbacks:
                callback.on_epoch_end(epoch_ix=epoch_ix, history=history)
                stop_training_flags = [callback.stop_training for callback in callbacks]
                stop_training = stop_training | (sum(stop_training_flags) > 0)
            if stop_training:
                print("Training stopped early. Epoch:", epoch_ix)
                break

        for callback in callbacks:
            callback.on_train_end(epoch_ix=epoch_ix, history=history)
        return history

    @torch.no_grad()
    def design_molecules(
        self,
        n_batches: int,
        batch_size: int,
        temperature: float,
        token2label: Dict[str, int],
    ):
        raise NotImplementedError
