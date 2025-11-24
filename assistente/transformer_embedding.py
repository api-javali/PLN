import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TransformerEmbedding:
    """Sistema de embeddings usando modelos de transformers (Sentence Transformers)"""
    
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        """
        Inicializa o modelo de transformers
        
        Args:
            model_name: Nome do modelo Sentence Transformers
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        
        # Stopwords em português (mesmo do Word2Vec)
        self.stopwords = [
            'a', 'as', 'e', 'o', 'os', 'da', 'de', 'do', 'um', 'uma',
            'em', 'no', 'na', 'nos', 'nas', 'ao', 'aos', 'à', 'às',
            'por', 'para', 'com', 'sem', 'sob', 'sobre',
            'que', 'qual', 'quais', 'quando', 'onde', 'como',
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
        """Pré-processa o texto (igual ao Word2Vec)"""
        text = text.lower().strip()
        text = re.sub(r'[^A-Za-zÀ-Ýà-ý\s]', '', text)
        words = text.split()
        words = [w for w in words if w and w not in self.stopwords]
        return ' '.join(words)  # Retorna string para Sentence Transformers
    
    def build_vocab(self, sentences):
        """Não precisa treinar, apenas carrega o modelo"""
        print(f"\n🤖 Carregando modelo Transformer: {self.model_name}")
        print(f"   • Dimensão dos vetores: {self.vector_size}")
        # Sentence Transformers já vem pré-treinado
    
    def text_to_vector(self, text):
        """Converte texto para vetor usando Sentence Transformers"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return np.zeros(self.vector_size)
        
        # Gera embedding da sentença
        embedding = self.model.encode([processed_text])[0]
        return np.array(embedding)
    
    def cosine_similarity(self, vec1, vec2):
        """Calcula similaridade cosseno"""
        return float(cosine_similarity([vec1], [vec2])[0][0])
    
    def most_similar(self, word, topn=5):
        """Não implementado para transformers (foco em sentenças)"""
        return []
    
    def get_vector(self, word):
        """Retorna vetor de uma palavra (aproximado)"""
        return self.model.encode([word])[0] if word else None