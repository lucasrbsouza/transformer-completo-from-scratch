import numpy as np

class LayerNorm:
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + self.epsilon)

class FeedForwardNetwork:
    def __init__(self, d_model: int, d_ff: int):
        np.random.seed(42)
        self.w1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2
    