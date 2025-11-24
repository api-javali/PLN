# assistente/transformer_embedding.py

import re
import numpy as np
from numpy import dot
from numpy.linalg import norm

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers não instalado. Execute: pip install transformers torch")


class TransformerEmbedding:
    """Sistema de embeddings usando modelos Transformer (BERT, RoBERTa, etc.)"""
    
    def __init__(self, model_name='neuralmind/bert-base-portuguese-cased'):
        """
        Inicializa o modelo Transformer
        
        Args:
            model_name: Nome do modelo HuggingFace
                - 'neuralmind/bert-base-portuguese-cased' (BERT PT-BR)
                - 'bert-base-multilingual-cased' (BERT Multilingual)
                - 'xlm-roberta-base' (XLM-RoBERTa)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers não disponível. Instale com: pip install transformers torch")
        
        self.model_name = model_name
        self.vector_size = 768  # BERT padrão usa 768 dimensões
        
        print(f"\n🤖 Carregando modelo Transformer: {model_name}")
        
        # Carrega tokenizer e modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Coloca modelo em modo de avaliação (desativa dropout, etc.)
        self.model.eval()
        
        # Configura dispositivo (CPU ou GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"✅ Modelo carregado no dispositivo: {self.device}")
        print(f"📏 Dimensão dos embeddings: {self.vector_size}")
        
        # Stopwords em português (mesmo do Word2Vec)
        self.stopwords = [
            'a', 'as', 'e', 'o', 'os', 'da', 'de', 'do', 'um', 'uma',
            'em', 'no', 'na', 'nos', 'nas', 'ao', 'aos', 'à', 'às',
            'por', 'para', 'com', 'sem', 'sob', 'sobre',
            'que', 'qual', 'quais', 'quando', 'onde', 'como', 'quero', 'querer', 'quer',
            'me', 'mim', 'te', 'ti', 'se', 'si', 'lhe', 'nos', 'vos',
            'eu', 'tu', 'ele',
            'olá', 'ola', 'oi', 'hey', 'ei', 'bom', 'dia', 'tarde', 'noite',
            'favor', 'please', 'obrigado', 'obrigada', 'valeu',
            'ok', 'tudo', 'bem', 'legal', 'show', 'beleza', 'massa',
            'sim', 'não', 'nao', 'claro', 'certamente', 'com certeza',
            'desculpe', 'desculpa', 'perdão', 'perdao', 'ops', 'opa',
            'por favor', 'agradeço', 'agradeco'
        ]
    
    def preprocess_text(self, text):
        """
        Pré-processa o texto (compatível com Word2Vec)
        
        Args:
            text: String de entrada
        
        Returns:
            Lista de palavras normalizadas
        """
        # Converte para minúsculas
        text = text.lower().strip()
        
        # Remove pontuação, mantém apenas letras (incluindo acentos)
        text = re.sub(r'[^A-Za-zÀ-Ýà-ý\s]', '', text)
        
        # Tokeniza em palavras
        words = text.split()
        
        # Remove stopwords
        words = [w for w in words if w and w not in self.stopwords]
        
        return words
    
    def text_to_vector(self, text):
        """
        Converte texto para vetor usando BERT embeddings
        
        Args:
            text: String de entrada
        
        Returns:
            Numpy array com vetor do documento (768 dimensões)
        """
        if not text or not text.strip():
            return np.zeros(self.vector_size)
        
        # Tokeniza com BERT tokenizer
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Gera embeddings (sem calcular gradientes)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Usa [CLS] token embedding (primeira posição) como representação da sentença
        # Alternativa: usar mean pooling de todos os tokens
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return cls_embedding
    
    def cosine_similarity(self, vec1, vec2):
        """
        Calcula similaridade cosseno entre dois vetores
        
        Args:
            vec1, vec2: Numpy arrays
        
        Returns:
            Float entre -1 e 1 (1 = muito similar, 0 = ortogonal, -1 = oposto)
        """
        norm1 = norm(vec1)
        norm2 = norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot(vec1, vec2) / (norm1 * norm2))
    
    def build_vocab(self, sentences):
        """
        Método compatível com Word2Vec (não faz nada no Transformer)
        BERT já vem pré-treinado, não precisa treinar vocabulário
        
        Args:
            sentences: Lista de listas de palavras (ignorado)
        """
        print(f"\n✅ Transformer pré-treinado - vocabulário já disponível")
        vocab_size = self.tokenizer.vocab_size
        print(f" • Vocabulário: {vocab_size:,} tokens")
        print(f" • Modelo: {self.model_name}")
    
    def most_similar(self, word, topn=5):
        """
        Método compatível com Word2Vec (não implementado para Transformer)
        BERT trabalha com contexto completo, não palavras isoladas
        """
        print(f"⚠️ most_similar() não disponível para Transformers")
        return []
    
    def get_vector(self, word):
        """
        Retorna embedding de uma palavra (com contexto limitado)
        
        Args:
            word: Palavra
        
        Returns:
            Numpy array ou None
        """
        return self.text_to_vector(word)
    
    def save_model(self, filepath):
        """Transformers não precisam ser salvos localmente (usa HuggingFace Hub)"""
        print(f"✅ Modelo Transformer não precisa ser salvo (usa HuggingFace Hub)")
    
    def load_model(self, filepath):
        """Transformers são carregados do HuggingFace Hub no __init__"""
        print(f"✅ Modelo Transformer já carregado do HuggingFace Hub")


# Teste rápido
if __name__ == "__main__":
    print("🧪 Testando Transformer Embedding...")
    
    try:
        # Cria embedding com BERT PT-BR
        bert = TransformerEmbedding(model_name='neuralmind/bert-base-portuguese-cased')
        
        # Testa conversão de texto
        test_texts = [
            "mude a cor do fundo para azul",
            "altere a cor de fundo para vermelho",
            "aumente o tamanho da fonte"
        ]
        
        print("\n📊 Gerando embeddings...")
        vectors = []
        for text in test_texts:
            vec = bert.text_to_vector(text)
            vectors.append(vec)
            print(f" • '{text}'")
            print(f"   Shape: {vec.shape}, Norm: {norm(vec):.2f}")
        
        # Testa similaridade
        print("\n🔍 Calculando similaridades...")
        sim_12 = bert.cosine_similarity(vectors[0], vectors[1])
        sim_13 = bert.cosine_similarity(vectors[0], vectors[2])
        
        print(f" • 'mude...azul' vs 'altere...vermelho': {sim_12:.4f}")
        print(f" • 'mude...azul' vs 'aumente...fonte': {sim_13:.4f}")
        
        print("\n✅ Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
