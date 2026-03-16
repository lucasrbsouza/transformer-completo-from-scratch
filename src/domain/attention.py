import numpy as np
from typing import Optional

class ScaledDotProductAttention:
    def __init__(self, d_model: int):
        self.d_model = d_model
        
        np.random.seed(42)
        self.w_q = np.random.randn(d_model, d_model) * 0.01
        self.w_k = np.random.randn(d_model, d_model) * 0.01
        self.w_v = np.random.randn(d_model, d_model) * 0.01

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, q_input: np.ndarray, k_v_input: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        q = q_input @ self.w_q
        k = k_v_input @ self.w_k
        v = k_v_input @ self.w_v

        k_t = np.swapaxes(k, -2, -1)
        scores = (q @ k_t) / np.sqrt(self.d_model)
        
        if mask is not None:
            scores = scores + mask
            
        attention_weights = self._softmax(scores)
        return attention_weights @ v