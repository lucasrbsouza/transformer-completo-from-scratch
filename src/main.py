import numpy as np
from infrastructure.embeddings import TokenEmbedding, PositionalEncoding
from domain.transformer import Transformer

def main():
    print("=== Laboratório 4: O Transformer Completo Fim-a-Fim ===\n")
    
    vocab = {
        "<START>": 0, "Thinking": 1, "Machines": 2, 
        "máquinas": 3, "pensantes": 4, "<EOS>": 5
    }
    vocab_size = len(vocab)
    
    d_model = 64
    d_ff = 256
    num_layers = 6
    
    embedding = TokenEmbedding(vocab_size, d_model)
    pos_encoding = PositionalEncoding(d_model)
    transformer = Transformer(num_layers, d_model, d_ff, vocab_size)
    
    encoder_input_words = ["Thinking", "Machines"]
    encoder_input_ids = [vocab[w] for w in encoder_input_words]
    
    print(f"-> Entrada do Encoder: {encoder_input_words}")
    
    enc_embed = embedding.forward(encoder_input_ids)
    enc_input = pos_encoding.forward(enc_embed)
    
    memory_z = transformer.encode(enc_input)
    print(f"-> Memória do Encoder (Z) gerada com shape: {memory_z.shape}\n")

if __name__ == "__main__":
    main()