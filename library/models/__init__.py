from .lstm import LSTM  # noqa: F401

__MODEL_NAMES__ = {
    "lstm": LSTM,
}


def get_chemical_language_model(model_name: str):
    clm = __MODEL_NAMES__.get(model_name, None)
    if clm is None:
        raise ValueError(f"Unknown model name {model_name}")
    return clm
