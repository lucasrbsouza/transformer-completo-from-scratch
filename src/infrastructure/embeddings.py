import numpy as np
from typing import List

class TokenEmbedding:
    def __init__(self, vocab_size: int, d_model: int):
        self.d_model = d_model
        # Inicialização simulada da matriz de vocabulário
        np.random.seed(42)
        self._weights = np.random.randn(vocab_size, d_model)

    def forward(self, token_ids: List[int]) -> np.ndarray:
        embeddings = self._weights[token_ids] * np.sqrt(self.d_model)
        return np.expand_dims(embeddings, axis=0)

class PositionalEncoding:
    def __init__(self, d_model: int, max_seq_len: int = 100):
        self.pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[1]
        
        return x + self.pe[:seq_len, :]