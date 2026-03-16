import numpy as np
from domain.attention import ScaledDotProductAttention
from domain.sublayers import LayerNorm, FeedForwardNetwork

class DecoderBlock:
    def __init__(self, d_model: int, d_ff: int):
        self.masked_attention = ScaledDotProductAttention(d_model)
        self.norm1 = LayerNorm()
        
        self.cross_attention = ScaledDotProductAttention(d_model)
        self.norm2 = LayerNorm()
        
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm3 = LayerNorm()

    def forward(self, y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> np.ndarray:
        y_att = self.masked_attention.forward(q_input=y, k_v_input=y, mask=mask)
        y_norm1 = self.norm1.forward(y + y_att)
        
        c_att = self.cross_attention.forward(q_input=y_norm1, k_v_input=z, mask=None)
        y_norm2 = self.norm2.forward(y_norm1 + c_att)
        
        y_ffn = self.ffn.forward(y_norm2)
        out = self.norm3.forward(y_norm2 + y_ffn)
        
        return out

class TransformerDecoder:
    def __init__(self, num_layers: int, d_model: int, d_ff: int, vocab_size: int):
        self.layers = [DecoderBlock(d_model, d_ff) for _ in range(num_layers)]
        
        np.random.seed(42)
        self.w_linear = np.random.randn(d_model, vocab_size) * 0.01
        self.b_linear = np.zeros(vocab_size)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            y = layer.forward(y, z, mask)
            
        logits = y @ self.w_linear + self.b_linear
        
        return self._softmax(logits)