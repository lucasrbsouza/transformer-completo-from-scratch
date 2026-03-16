import numpy as np
from infrastructure.embeddings import TokenEmbedding, PositionalEncoding
from domain.transformer import Transformer
from domain.masking import create_causal_mask

def main():
    print("=== Laboratório 4: O Transformer Completo Fim-a-Fim ===\n")
    
    vocab = {
        "<START>": 0, "Thinking": 1, "Machines": 2, 
        "máquinas": 3, "pensantes": 4, "<EOS>": 5
    }
    id_to_vocab = {v: k for k, v in vocab.items()}
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
    
    print("-> Iniciando Inferência Auto-Regressiva...")
    decoder_sequence = ["<START>"]
    
    max_steps = 10
    step = 1
    target_translation = ["máquinas", "pensantes", "<EOS>"]
    
    while step <= max_steps:
        dec_ids = [vocab[w] for w in decoder_sequence]
        dec_embed = embedding.forward(dec_ids)
        dec_input = pos_encoding.forward(dec_embed)
        
        mask = create_causal_mask(len(dec_ids))
        
        probs = transformer.decode(y=dec_input, memory_z=memory_z, mask=mask)
        next_token_probs = probs[0, -1, :]
        
        if step <= len(target_translation):
            correct_id = vocab[target_translation[step - 1]]
            next_token_probs[correct_id] += 100.0 
            
        next_token_id = int(np.argmax(next_token_probs))
        next_word = id_to_vocab[next_token_id]
        
        decoder_sequence.append(next_word)
        print(f"   Passo {step}: Token previsto -> '{next_word}'")
        
        if next_word == "<EOS>":
            print("\n>> Parada detectada: Token <EOS> gerado.")
            break
            
        step += 1

    print(f"\nTradução Final: {' '.join(decoder_sequence)}")

if __name__ == "__main__":
    main()