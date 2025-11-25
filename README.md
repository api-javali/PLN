# PLN - Assistente Virtual Inteligente

Este projeto implementa um assistente virtual inteligente que utiliza Processamento de Linguagem Natural (PLN) para interpretar comandos em linguagem natural e modificar uma interface web em tempo real. O sistema combina embeddings de palavras (Word2Vec e Transformer BERT), uma rede neural MLP com backpropagation e similaridade cosseno para decisões híbridas.

## 🚀 Funcionalidades

- **Interpretação de Comandos**: Reconhece comandos como "mude a cor do fundo para azul" ou "vá para a página sobre".
- **Modificação de Interface**: Altera cores, tamanho de fonte, posição de componentes (menu, header) e navegação entre páginas.
- **Modelos de Embedding**: Suporte a Word2Vec e Transformer (BERT) para geração de vetores.
- **Classificação MLP**: Rede neural treinada com backpropagation para predição de ações.
- **Sistema Híbrido**: Combina MLP, similaridade cosseno e validação semântica para decisões inteligentes.
- **Correção Ortográfica**: Usa distância de Levenshtein para corrigir erros comuns.
- **Comparação de Modelos**: Interface para comparar predições entre Word2Vec e Transformer.


## 🛠 Tecnologias

- **Backend**: Python 3.8+, Flask
- **ML/PLN**: Gensim (Word2Vec), Transformers (BERT), NumPy
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Outros**: JSON para dados de treinamento

## 📦 Instalação

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/api-javali/PLN.git
   
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute o servidor**:
   ```bash
   python app.py
   ```

4. **Acesse**: Abra `http://localhost:5000` no navegador.

## 🎯 Uso

- Digite comandos no chat, como "mude o fundo para vermelho" ou "aumente a fonte".
- Use os botões de exemplo para testar.
- Alterne entre modelos (Word2Vec/Transformer) no seletor.
- Compare modelos digitando uma frase no campo de comparação.

### Exemplos de Comandos
- "mude a cor do fundo para azul"
- "aumente o tamanho da fonte"
- "vá para a página sobre"
- "mova o menu para a esquerda"
- "redefinir layout"

## 📁 Estrutura do Projeto

```
.
├── app.py                          # Servidor Flask principal
├── assistente/
│   ├── virtual_assistant.py        # Lógica do assistente
│   ├── word2vec.py                 # Embedding Word2Vec
│   ├── transformer_embedding.py    # Embedding Transformer
│   ├── classifier.py               # MLP com backpropagation
│   └── model_comparison.py         # Comparação de modelos
├── static/
│   ├── index.js                    # JavaScript frontend
│   └── styles.css                  # Estilos CSS
├── templates/
│   └── index.html                  # Template HTML
├── training_data.json              # Dados de treinamento
├── requirements.txt                # Dependências Python
└── README.md                       # Este arquivo
```

## 🔧 Como Funciona

1. **Pré-processamento**: Texto é tokenizado, limpo e corrigido ortograficamente.
2. **Embeddings**: Palavras são convertidas em vetores (Word2Vec ou BERT).
3. **Classificação**: MLP prevê a ação baseada nos vetores.
4. **Decisão Híbrida**: Combina MLP, similaridade e validação para executar ações.
5. **Execução**: Interface é atualizada via JavaScript.

