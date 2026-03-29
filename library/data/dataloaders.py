# %%
from typing import Dict, Tuple

import torch

from library.data import data_utils
from library.smiles import smiles_utils


class CLMLoader(torch.utils.data.Dataset):
    def __init__(self, label_encoded_molecules: torch.LongTensor):
        self.label_encoded_molecules = label_encoded_molecules

    def __len__(self) -> int:
        return self.label_encoded_molecules.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.LongTensor, torch.LongTensor]:
        molecule = self.label_encoded_molecules[idx, :]
        X = molecule[:-1]
        y = molecule[1:]
        return X, y


def get_dataloader(
    path_to_data: str,
    batch_size: int,
    sequence_length: int,
    token2label: Dict[str, int],
    num_workers: int = 8,
    shuffle: bool = True,
    fraction_of_dataset: float = 1.0,
    smiles_enumeration: bool = False,
) -> torch.utils.data.DataLoader:
    with open(path_to_data, "r") as f:
        molecules = [line.strip() for line in f.readlines()]

    n_molecules = int(fraction_of_dataset * len(molecules))
    molecules = molecules[:n_molecules]

    if smiles_enumeration:
        molecules = [smiles_utils.enumerate_smiles(smiles) for smiles in molecules]
        molecules = [smiles for smiles_list in molecules for smiles in smiles_list]

    molecules_tensor = data_utils.molecules_to_tensor(
        molecules, sequence_length, token2label, space_separated=False
    )

    return torch.utils.data.DataLoader(
        CLMLoader(
            molecules_tensor,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    import json

    with open("./data/token2label.json", "r") as f:
        token2label = json.load(f)

    train_loader = get_dataloader(
        "./datasets/chemblv31/train.smiles",
        batch_size=16,
        sequence_length=100,
        token2label=token2label,
        num_workers=8,
        shuffle=True,
    )
    val_loader = get_dataloader(
        "./datasets/chemblv31/valid.smiles",
        batch_size=16,
        sequence_length=100,
        token2label=token2label,
        num_workers=8,
        shuffle=False,
    )
