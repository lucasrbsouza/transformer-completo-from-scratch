# O Transformer Completo "From Scratch" - Lab 04

**Instituição:** iCEV - Instituto de Ensino Superior

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1

**Professor:** Prof. Dimmy Magalhães

**Aluno:** Lucas Souza

## 1. Objetivo do Projeto

Este repositório contém uma implementação completa e funcional do modelo **Transformer** (arquitetura Encoder-Decoder), construída integralmente "do zero" utilizando apenas `Python` e `NumPy`, sem dependências de frameworks de deep learning.

**Funcionalidades principais:**
- ✅ Inferência fim-a-fim com entrada de texto
- ✅ Geração auto-regressiva com detecção de fim de sequência (EOS)
- ✅ Implementação de Positional Encoding
- ✅ Mecanismos de Self-Attention e Cross-Attention
- ✅ Camadas de normalização e redes feed-forward

## 2. Estrutura do Projeto

```
transformer-completo-from-scratch/
├── src/
│   ├── domain/                  # Núcleo matemático do Transformer
│   │   ├── attention.py         # Implementação do mecanismo de atenção
│   │   ├── encoder.py           # Estrutura do Transformer Encoder
│   │   ├── decoder.py           # Estrutura do Transformer Decoder
│   │   ├── masking.py           # Máscaras usadas em attention (padding e causal)
│   │   ├── sublayers.py         # Subcamadas do modelo (LayerNorm, FeedForward, etc.)
│   │   └── transformer.py       # Modelo Transformer completo (orquestra encoder/decoder)
│   │
│   ├── infrastructure/          # Componentes ligados à representação dos dados
│   │   └── embeddings.py        # TokenEmbedding e PositionalEncoding
│   │
│   └── main.py                  # Ponto de entrada da aplicação e execução do modelo
│
├── .gitignore
├── README.md
```

## 3. Engenharia de Software Aplicada


O projeto abandona a organização procedural comum e adota uma arquitetura mais limpa, tendo referencias em **Clean Architecture** e nos princípios **SOLID**, favorecendo modularidade, testabilidade e clareza conceitual.

* **`src/domain/`**: Contém o núcleo matemático do modelo. Componentes fundamentais como `LayerNorm` e `FeedForwardNetwork` funcionam como unidades independentes, enquanto `TransformerEncoder` e `TransformerDecoder` coordenam as pilhas de camadas. A classe `Transformer` atua como uma **fachada**, oferecendo uma interface simples para interação com todo o sistema.

* **`src/infrastructure/`**: Responsável por detalhes de implementação ligados à representação dos dados. Aqui ficam módulos como `TokenEmbedding`, que converte tokens em vetores, e `PositionalEncoding`, que injeta informação posicional por meio de funções trigonométricas.

* **`src/main.py`**: Atua como ponto de orquestração da aplicação, encarregado da injeção de dependências e do controle do fluxo de execução, incluindo o laço auto-regressivo utilizado durante a geração de sequências.


## 4. Fluxo de Inferência (Auto-Regressivo)

```
Entrada: "Thinking Machines"
    ↓
[1] Embedding + Positional Encoding
    ↓
[2] Encoder (processamento bidirecional)
    ↓
Inicializar Decoder com <START>
    ↓
[3] LOOP Auto-Regressivo:
    ├─ Masked Self-Attention (previne vazamento do futuro)
    ├─ Cross-Attention (integra contexto do Encoder)
    ├─ Feed-Forward Network (transformação não-linear)
    ├─ Layer Normalization (estabilização)
    ├─ Predição do próximo token
    └─ Se token = <EOS>, encerrar; senão, repetir com novo contexto
    ↓
Saída: Sequência gerada
```

**Passos em detalhes:**

1. **Embedding e Encoding Posicional**: A frase de entrada *"Thinking Machines"* é convertida em embeddings e enriquecida com informação de posição.

2. **Processamento do Encoder**: A pilha do Encoder processa a entrada bidirecionalmente, permitindo que cada token "veja" todos os outros. Gera a matriz de contexto `Z` que será reutilizada pelo Decoder.

3. **Inicialização do Decoder**: O Decoder começa com o token especial `<START>`.

4. **Geração Auto-Regressiva**: A cada passo:
   - **Masked Self-Attention**: Impede que o modelo veja tokens futuros (causalidade)
   - **Cross-Attention**: Integra informação do Encoder (`Z`) com o contexto atual
   - A rede prediz a próxima palavra e a realimenta no Decoder
   - Continua até gerar o token `<EOS>` (End-Of-Sequence)

## 5. Conceitos Matemáticos Chave

### Multi-Head Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
Múltiplas "cabeças" de atenção permitem capturar diferentes relações entre tokens.

### Positional Encoding
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Codificação trigonométrica que preserva informação relativa de posição.

### Masked Attention (Causalidade)
Durante decodificação, apenas tokens anteriores são visíveis, prevenindo vazamento de informação futura.

## 6. Como Executar

### Pré-requisitos
- Python 3.8+
- NumPy

### Instalação
```bash
# Clonar o repositório
git clone <https://github.com/lucasrbsouza/transformer-completo-from-scratch.git>
cd transformer-completo-from-scratch

# Instalar dependências (opcional - apenas NumPy se necessário)
pip install numpy
```

### Inferência
```bash
# Executar inferência fim-a-fim
python src/main.py
```

**Saída esperada:**
```
=== Laboratório 4: O Transformer Completo Fim-a-Fim ===

-> Entrada do Encoder: ['Thinking', 'Machines']
-> Memória do Encoder (Z) gerada com shape: (1, 2, 64)

-> Iniciando Inferência Auto-Regressiva...
   Passo 1: Token previsto -> 'máquinas'
   Passo 2: Token previsto -> 'pensantes'
   Passo 3: Token previsto -> '<EOS>'

>> Parada detectada: Token <EOS> gerado.

Tradução Final: <START> máquinas pensantes <EOS>
```

## 7. Componentes Principais

| Arquivo          | Responsabilidade                                                                                                           |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `attention.py`   | Implementação do mecanismo de atenção utilizado pelo modelo (self-attention e cross-attention).                            |
| `sublayers.py`   | Implementação das subcamadas fundamentais do Transformer, como **Layer Normalization** e a **Feed-Forward Network (MLP)**. |
| `masking.py`     | Geração das máscaras utilizadas na atenção, como **padding mask** e **causal mask** para o decoder.                        |
| `encoder.py`     | Implementação da pilha de camadas do **Transformer Encoder**, responsável por processar a sequência de entrada.            |
| `decoder.py`     | Implementação da pilha do **Transformer Decoder**, incluindo masked attention e cross-attention com o encoder.             |
| `transformer.py` | Definição do modelo Transformer completo, integrando encoder e decoder.                                                    |
| `embeddings.py`  | Implementação das representações vetoriais dos tokens (**Token Embeddings**) e da **Positional Encoding**.                 |
| `main.py`        | Ponto de entrada da aplicação, responsável por inicializar o modelo e executar o fluxo de inferência.                      |


## 8. Nota de Integridade e Uso de IA

"Partes geradas/complementadas com IA, revisadas por Lucas Souza".

A lógica matemática intrínseca e o empilhamento das redes seguiram rigorosamente as equações do roteiro fornecido pelo professor, sendo a Inteligência Artificial empregada estritamente como ferramenta de refatoração para adequação aos princípios de engenharia de software e processamento de linguagem natural.

## 9. Referências

- "Attention is All You Need" (Vaswani et al., 2017)
- Roteiro do Laboratório 04 - Prof. Dimmy Magalhães
- Materiais da disciplina de Tópicos em IA
