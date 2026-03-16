import numpy as np

def create_causal_mask(seq_len: int) -> np.ndarray:
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    causal_mask = np.zeros((seq_len, seq_len))
    causal_mask[mask] = -np.inf
    return causal_mask
