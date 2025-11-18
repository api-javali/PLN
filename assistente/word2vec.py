import re
import numpy as np
from numpy import dot, average
from numpy.linalg import norm
from gensim.models import Word2Vec


class Word2VecEmbedding:
    """Sistema de embeddings usando Word2Vec do Gensim"""
    
    def __init__(self, vector_size=50, window=3, min_count=1, epochs=50):
        """
        Inicializa o Word2Vec
        
        Args:
            vector_size: Dimensão dos vetores de palavra (50 é um bom compromisso)
            window: Janela de contexto (palavras antes/depois)
            min_count: Frequência mínima para incluir palavra no vocabulário
            epochs: Número de épocas de treinamento
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
        
        # Stopwords em português (expandido)
        self.stopwords = [
            'a', 'as', 'e', 'o', 'os', 'da', 'de', 'do', 'um', 'uma',
            'em', 'no', 'na', 'nos', 'nas', 'ao', 'aos', 'à', 'às',
            'por', 'para', 'com', 'sem', 'sob', 'sobre',
            'que', 'qual', 'quais', 'quando', 'onde', 'como', 'quero', 'querer', 'quer',
            'me', 'mim', 'te', 'ti', 'se', 'si', 'lhe', 'nos', 'vos',
            'eu', 'tu', 'ele',
            # Saudações e palavras irrelevantes para comandos
            'olá', 'ola', 'oi', 'hey', 'ei', 'bom', 'dia', 'tarde', 'noite',
            'favor', 'please', 'obrigado', 'obrigada', 'valeu',
            'ok', 'tudo', 'bem', 'legal', 'show', 'beleza', 'massa',
            'sim', 'não', 'nao', 'claro', 'certamente', 'com certeza',
            'desculpe', 'desculpa', 'perdão', 'perdao', 'ops', 'opa',
            'por favor', 'agradeço', 'agradeco'
        ]
    
    def preprocess_text(self, text):
        """
        Pré-processa o texto 
        
        Args:
            text: String de entrada
            
        Returns:
            Lista de palavras normalizadas
        """
        # 1. Converte para minúsculas
        text = text.lower().strip()
        
        # 2. Remove pontuação, mantém apenas letras (incluindo acentos)
        # Regex: [^A-Za-zÀ-Ýà-ý] remove tudo que não é letra
        text = re.sub(r'[^A-Za-zÀ-Ýà-ý\s]', '', text)
        
        # 3. Tokeniza em palavras
        words = text.split()
        
        # 4. Remove stopwords
        words = [w for w in words if w and w not in self.stopwords]
        
        return words
    
    def build_vocab(self, sentences):
        """
        Treina o modelo Word2Vec com as sentenças de treinamento
        
        Args:
            sentences: Lista de listas de palavras
                      Ex: [['mude', 'cor', 'fundo'], ['aumente', 'fonte']]
        """
        print(f"\n📚 Treinando Word2Vec...")
        print(f"   • Sentenças: {len(sentences)}")
        print(f"   • Vector size: {self.vector_size}")
        print(f"   • Window: {self.window}")
        print(f"   • Epochs: {self.epochs}")
        
        # Treina o modelo Word2Vec usando configuração otimizada
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sample=1e-5,  # Downsampling de palavras frequentes
            epochs=self.epochs,
            workers=1,  # Número de threads
            sg=0  # 0 = CBOW, 1 = Skip-gram (CBOW é mais rápido e bom para datasets pequenos)
        )
        
        vocab_size = len(self.model.wv)
        print(f"   ✅ Vocabulário construído: {vocab_size} palavras únicas")
        
        # Mostra algumas palavras do vocabulário
        sample_words = list(self.model.wv.index_to_key)[:10]
        print(f"   📝 Amostra do vocabulário: {sample_words}")
    
    def text_to_vector(self, text):
        """
        Converte texto para vetor usando média dos word embeddings: average de word_embedding[:,i]
        
        Args:
            text: String de entrada
            
        Returns:
            Numpy array com vetor do documento
        """
        words = self.preprocess_text(text)
        
        if not words:
            # Se não há palavras válidas, retorna vetor zero
            return np.zeros(self.vector_size)
        
        # Coleta vetores das palavras que existem no vocabulário
        word_vectors = []
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        if not word_vectors:
            # Nenhuma palavra estava no vocabulário
            return np.zeros(self.vector_size)
        
        # Converte para numpy array
        word_embedding = np.array(word_vectors)
        
        # Calcula média de cada dimensão 
        doc_embedding = np.zeros(self.vector_size)
        for i in range(self.vector_size):
            doc_embedding[i] = average(word_embedding[:, i])
        
        return doc_embedding
    
    def cosine_similarity(self, vec1, vec2):
        """
        Calcula similaridade cosseno entre dois vetores
        Fórmula: dot(v1, v2) / (norm(v1) * norm(v2))
        
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
    
    def most_similar(self, word, topn=5):
        """
        Encontra palavras mais similares a uma palavra dada
        Usa o método nativo do Gensim
        
        Args:
            word: Palavra de referência
            topn: Número de palavras similares a retornar
            
        Returns:
            Lista de tuplas (palavra, similaridade)
        """
        if not self.model or word not in self.model.wv:
            return []
        
        return self.model.wv.most_similar(word, topn=topn)
    
    def get_vector(self, word):
        """
        Retorna o vetor de uma palavra específica
        
        Args:
            word: Palavra
            
        Returns:
            Numpy array ou None se palavra não existe
        """
        word = word.lower()
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        return None
    
    def save_model(self, filepath):
        """Salva o modelo treinado"""
        if self.model:
            self.model.save(filepath)
            print(f"✅ Modelo salvo em {filepath}")
    
    def load_model(self, filepath):
        """Carrega modelo previamente treinado"""
        self.model = Word2Vec.load(filepath)
        self.vector_size = self.model.vector_size
        print(f"✅ Modelo carregado de {filepath}")
        print(f"   • Vocabulário: {len(self.model.wv)} palavras")


# Teste rápido se executado diretamente
if __name__ == "__main__":
    print("🧪 Testando Word2Vec...")
    
    # Dados de exemplo
    sentences = [
        ['mude', 'cor', 'fundo', 'azul'],
        ['altere', 'cor', 'fundo', 'vermelho'],
        ['aumente', 'tamanho', 'fonte'],
        ['diminua', 'fonte'],
        ['navegue', 'página', 'inicial'],
        ['vá', 'página', 'contato']
    ]
    
    # Cria e treina
    w2v = Word2VecEmbedding(vector_size=32, epochs=100)
    w2v.build_vocab(sentences)
    
    # Testa conversão de texto
    test_text = "Olá, mude a cor do fundo para verde"
    vector = w2v.text_to_vector(test_text)
    print(f"\n📊 Vetor do texto: '{test_text}'")
    print(f"   Shape: {vector.shape}")
    print(f"   Primeiros 5 valores: {vector[:5]}")
    
    # Testa similaridade
    if 'mude' in w2v.model.wv and 'altere' in w2v.model.wv:
        v1 = w2v.get_vector('mude')
        v2 = w2v.get_vector('altere')
        sim = w2v.cosine_similarity(v1, v2)
        print(f"\n🔍 Similaridade entre 'mude' e 'altere': {sim:.4f}")
    
    # Palavras mais similares
    if 'mude' in w2v.model.wv:
        similar = w2v.most_similar('mude', topn=3)
        print(f"\n💡 Palavras similares a 'mude':")
        for word, score in similar:
            print(f"   • {word}: {score:.4f}")
