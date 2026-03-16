import numpy as np
from domain.attention import ScaledDotProductAttention
from domain.sublayers import LayerNorm, FeedForwardNetwork

class EncoderBlock:
    def __init__(self, d_model: int, d_ff: int):
        self.self_attention = ScaledDotProductAttention(d_model)
        self.norm1 = LayerNorm()
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = LayerNorm()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_att = self.self_attention.forward(q_input=x, k_v_input=x)
        x_norm1 = self.norm1.forward(x + x_att)
        
        x_ffn = self.ffn.forward(x_norm1)
        z_out = self.norm2.forward(x_norm1 + x_ffn)
        
        return z_out
    