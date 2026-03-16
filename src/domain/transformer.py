import numpy as np
from domain.encoder import EncoderBlock
from domain.decoder import TransformerDecoder

class TransformerEncoder:
    def __init__(self, num_layers: int, d_model: int, d_ff: int):
        self.layers = [EncoderBlock(d_model, d_ff) for _ in range(num_layers)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

class Transformer:
    def __init__(self, num_layers: int, d_model: int, d_ff: int, vocab_size: int):
        if num_layers <= 0 or d_model <= 0:
            raise ValueError("Parâmetros da arquitetura devem ser positivos.")
            
        self.encoder = TransformerEncoder(num_layers, d_model, d_ff)
        self.decoder = TransformerDecoder(num_layers, d_model, d_ff, vocab_size)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Processa a entrada e gera a matriz de memória rica (Z)."""
        return self.encoder.forward(x)

    def decode(self, y: np.ndarray, memory_z: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Gera as probabilidades do próximo token baseando-se no contexto (Y) e na memória (Z)."""
        return self.decoder.forward(y, memory_z, mask)