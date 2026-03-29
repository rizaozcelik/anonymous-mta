from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, SimDivFilters
from rdkit.Chem import rdMolDescriptors
from scipy import stats

RDLogger.DisableLog("rdApp.*")


def compute_success_rate(
    df_canonical_designs: pd.DataFrame,
    descriptor_names: List[str],
    descriptor_thresholds: List[float],
    comparisons: List[str],
) -> Dict[str, float]:
    num_designs = df_canonical_designs.shape[0]
    num_invalids = int(df_canonical_designs["is_novel"].isna().sum())
    num_valids = num_designs - num_invalids

    df_novel = df_canonical_designs.fillna(0).query("is_novel > 0")
    df_novel_and_unique = df_novel.drop_duplicates("can_smiles")
    num_novel = df_novel_and_unique.shape[0]

    for task_idx, (descriptor_name, descriptor_threshold, comparison) in enumerate(
        zip(descriptor_names, descriptor_thresholds, comparisons)
    ):
        df_novel_and_unique[f"successful_design_task_{task_idx}"] = (
            df_novel_and_unique[descriptor_name] > descriptor_threshold
            if comparison == "greater"
            else df_novel_and_unique[descriptor_name] <= descriptor_threshold
        )

    df_novel_and_unique["successful_design"] = (
        df_novel_and_unique[
            [
                f"successful_design_task_{task_idx}"
                for task_idx in range(len(descriptor_names))
            ]
        ]
        .all(axis=1)
        .astype(int)
    )
    num_successful_novel_designs = int(df_novel_and_unique["successful_design"].sum())
    num_failed_designs = num_designs - num_successful_novel_designs

    success_metrics = {
        "validity": num_valids / num_designs,
        "novelty": num_novel / num_designs,
        "num-novel-successful-designs": num_successful_novel_designs,
        "success-rate": num_successful_novel_designs / num_designs,
        "novel-success-rate": num_successful_novel_designs / num_novel,
        "num-failed-designs": num_failed_designs,
        "fail-rate": num_failed_designs / num_designs,
        "novel-fail-rate": num_failed_designs / num_novel,
    }

    df_successful_designs = df_novel_and_unique[
        df_novel_and_unique["successful_design"] == 1
    ]
    successful_diversity = compute_diversity(
        designs_batch=df_successful_designs["can_smiles"].tolist(),
        distance_threshold=0.65,
    )
    successful_diversity = {
        f"successful-{metric_name}": value
        for metric_name, value in successful_diversity.items()
    }
    return {**success_metrics, **successful_diversity}


def compute_ks_distances(
    df_canonical_designs: pd.DataFrame,
    dataset_name: str,
    setup_idx: int,
    descriptor_names: List[str],
) -> Dict[str, float]:
    computing_pt_distance = "chemblv33" in dataset_name
    if computing_pt_distance:
        path_to_dataset_descriptor = "./data/chemblv33/descriptors/"
        fold_names = ["train"]
    else:
        path_to_dataset_descriptor = (
            f"./data/{dataset_name}/setup-{setup_idx}/descriptors/"
        )
        fold_names = ["train", "test"]
    setup_scores = dict()
    for descriptor_name in descriptor_names:
        design_descriptors = df_canonical_designs.dropna(subset=descriptor_name)[
            descriptor_name
        ].values
        for fold_name in fold_names:
            dataset_descriptors = np.loadtxt(
                f"{path_to_dataset_descriptor}/{fold_name}/{descriptor_name}.txt"
            )
            dataset_descriptors = dataset_descriptors.reshape(-1, 1)

            ks_distance, p_val = stats.ks_2samp(
                design_descriptors, dataset_descriptors.flatten()
            )
            if computing_pt_distance:
                setup_scores[f"{descriptor_name}-chemblv33-{fold_name}-ks-distance"] = (
                    ks_distance
                )
                setup_scores[f"{descriptor_name}-chemblv33-{fold_name}-p-val"] = p_val
            else:
                setup_scores[f"{descriptor_name}-{fold_name}-ks-distance"] = ks_distance
                setup_scores[f"{descriptor_name}-{fold_name}-p-val"] = p_val

    return setup_scores


def compute_diversity(
    designs_batch: List[str], distance_threshold: float
) -> Dict[str, int]:
    substructures = dict()
    molecule_batch = [Chem.MolFromSmiles(design) for design in designs_batch]
    for mol in molecule_batch:
        morgan_vect = rdMolDescriptors.GetMorganFingerprint(mol, radius=2)
        non_zeros = morgan_vect.GetNonzeroElements()
        for key, value in non_zeros.items():
            if key not in substructures:
                substructures[key] = value
            else:
                substructures[key] += value

    no_substructures = len(substructures)

    bit_vects = [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        for mol in molecule_batch
    ]

    lp = SimDivFilters.LeaderPicker()
    picks = lp.LazyBitVectorPick(bit_vects, len(bit_vects), distance_threshold)
    no_clusters = len(picks)

    return {"no-substructures": no_substructures, "no-clusters": no_clusters}
