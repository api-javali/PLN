import json
import numpy as np
import threading
from assistente.transformer_embedding import TransformerEmbedding
from assistente.word2vec import Word2VecEmbedding
from assistente.classifier import MLPClassifier


class VirtualAssistant:
    def __init__(self):
        # Carrega ambos os embeddings na inicialização
        self.word2vec_embedding = Word2VecEmbedding(vector_size=50, window=3, epochs=100)
        self.transformer_embedding = TransformerEmbedding(model_name='paraphrase-multilingual-MiniLM-L12-v2')
        
        # Define o padrão
        self.embedding_model_type = 'word2vec'
        self.word_embedding = self.word2vec_embedding
        
        self.mlp = None
        self.classifiers = {}         # cache de MLP por modelo: 'word2vec' | 'transformer'
        self.command_vectors = {}     # cache de vetores dos comandos por modelo
        self.training_threads = {}
        self.training_lock = threading.Lock()
        
        self.label_encoder = {}
        self.label_decoder = {}
        self.commands_data = []
        
        # Estado da interface
        self.current_state = {
            'background_color': '#ffffff',
            'text_color': '#000000',
            'font_size': 16,
            'menu_position': 'center',
            'current_page': 'home',
            'header_position': 'top'
        }
        
        self.chat_history = []
        self.last_suggestion = None
        self.waiting_confirmation = False
        
        # Thresholds
        self.confidence_threshold = 0.60
        self.similarity_threshold = 0.45
        self.high_similarity_threshold = 0.60
        
        # Carrega dados
        self.load_training_data()
        # Prepara actions/labels
        self._prepare_labels()
        # Prepara modelos e classifiers em background (para troca rápida)
        self.prepare_models_background()
        # Aguarda treinamentos mínimos ou carrega classifier default
        # Se quiser iniciar com modelo alternativo, pode trocar aqui
        if 'word2vec' in self.classifiers:
            self.mlp = self.classifiers['word2vec']
        else:
            # se não estiver pronto, treinamos o atual sincronamente
            self._train_for_model('word2vec')
            self.mlp = self.classifiers.get('word2vec')
    
    def _prepare_labels(self):
        actions = sorted(set(cmd['action'] for cmd in self.commands_data))
        self.label_encoder = {action: idx for idx, action in enumerate(actions)}
        self.label_decoder = {idx: action for action, idx in self.label_encoder.items()}
    
    def prepare_models_background(self):
        """Inicia threads para pré-treinar MLPs e pré-computar vetores"""
        for model in ['word2vec', 'transformer']:
            t = threading.Thread(target=self._train_for_model, args=(model,), daemon=True)
            self.training_threads[model] = t
            t.start()
    
    def _compute_command_vectors(self, model_type):
        """Pré-computa e normaliza vetores dos comandos para busca rápida"""
        if model_type in self.command_vectors:
            return self.command_vectors[model_type]
        
        embed = self.word2vec_embedding if model_type == 'word2vec' else self.transformer_embedding
        vectors = []
        for cmd in self.commands_data:
            vec = embed.text_to_vector(cmd['text'])
            if vec is None:
                vec = np.zeros(embed.vector_size)
            vec = np.array(vec, dtype=float)
            norm = np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else 1.0
            vectors.append(vec / norm)
        if vectors:
            arr = np.vstack(vectors)
        else:
            arr = np.zeros((0, embed.vector_size))
        self.command_vectors[model_type] = arr
        return arr
    
    def _prepare_dataset_for_model(self, model_type):
        """Prepares X, y using cached command vectors or computing them"""
        # Build vocab if needed (word2vec)
        texts = [cmd['text'] for cmd in self.commands_data]
        if model_type == 'word2vec':
            tokenized_texts = [self.word2vec_embedding.preprocess_text(t) for t in texts]
            # build vocab only once safely
            self.word2vec_embedding.build_vocab(tokenized_texts)
        
        # Ensure command vectors computed
        vectors = self._compute_command_vectors(model_type)
        X = [v.tolist() for v in vectors]
        
        actions = sorted(set(cmd['action'] for cmd in self.commands_data))
        y = []
        for cmd in self.commands_data:
            label_vector = [0.0] * len(actions)
            label_vector[actions.index(cmd['action'])] = 1.0
            y.append(label_vector)
        return X, y, vectors.shape[1] if vectors.size else (self.word2vec_embedding.vector_size if model_type=='word2vec' else self.transformer_embedding.vector_size)
    
    def _train_for_model(self, model_type, epochs=200):
        """Treina classifier para um modelo específico e armazena no cache."""
        with self.training_lock:
            # Evita retreinar se já existe
            if model_type in self.classifiers:
                return self.classifiers[model_type]
            print(f"\n🔧 Treinando classificador para: {model_type}")
            X, y, input_size = self._prepare_dataset_for_model(model_type)
            if not X or not y:
                print("⚠️ Dataset vazio ou insuficiente para treinar")
                return None
            hidden_size = min(48, max(12, input_size // 2))
            output_size = len(y[0])
            mlp = MLPClassifier(input_size, hidden_size, output_size, learning_rate=0.15)
            mlp.train(X, y, epochs=epochs)
            self.classifiers[model_type] = mlp
            # se o modelo atual é esse, atualiza self.mlp
            if self.embedding_model_type == model_type:
                self.mlp = mlp
            print(f"✅ Classificador treinado para: {model_type}")
            return mlp
    
    def switch_model(self, new_model, background=True):
        """Alterna entre modelos sem recriar objetos pesados.
        Se background=True, treina classificador em background se não existir.
        """
        if new_model not in ['word2vec', 'transformer']:
            raise ValueError("Modelo inválido")

        self.embedding_model_type = new_model
        self.word_embedding = self.word2vec_embedding if new_model == 'word2vec' else self.transformer_embedding

        # Ajuste de thresholds (opcional)
        if new_model == 'transformer':
            self.confidence_threshold = 0.55
            self.similarity_threshold = 0.40
            self.high_similarity_threshold = 0.55
        else:
            self.confidence_threshold = 0.60
            self.similarity_threshold = 0.45
            self.high_similarity_threshold = 0.60

        # Precompute vectors for fast similarity (sempre útil)
        self._compute_command_vectors(new_model)

        # Se já existe classificador treinado, use-o
        if new_model in self.classifiers:
            self.mlp = self.classifiers[new_model]
            return f"Modelo alterado para {new_model} (cache pronto)"

        # Não existe classificador pronto — não usar MLP até treinar
        self.mlp = None

        # Treina em background ou de forma síncrona
        if background:
            t = threading.Thread(target=self._train_for_model, args=(new_model,), daemon=True)
            t.start()
            self.training_threads[new_model] = t
            return f"Modelo alterado para {new_model}. Classificador treinando em background..."
        else:
            mlp = self._train_for_model(new_model)
            self.mlp = mlp
            return f"Modelo alterado para {new_model} (treinado agora)."
    
    def load_training_data(self):
        """Carrega dados de treinamento do arquivo JSON"""
        try:
            with open('training_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.commands_data = data.get('commands', [])
                print(f"✅ {len(self.commands_data)} comandos carregados do training_data.json")
        except FileNotFoundError:
            print("⚠️ Arquivo training_data.json não encontrado")
            self.commands_data = []
    
    def train_models(self):
        """Treina o modelo atual (compatibilidade backward)"""
        return self._train_for_model(self.embedding_model_type)
    
    def calculate_similarity(self, vec1, vec2):
        """Calcula similaridade cosseno entre dois vetores"""
        return self.word_embedding.cosine_similarity(vec1, vec2)
    
    def find_most_similar_command(self, user_input):
        """Encontra o comando mais similar no treinamento usando cache e produto escalar"""
        user_vec = self.word_embedding.text_to_vector(user_input)
        if user_vec is None:
            return None, 0.0
        user_vec = np.array(user_vec, dtype=float)
        norm = np.linalg.norm(user_vec) if np.linalg.norm(user_vec) != 0 else 1.0
        user_vec_norm = user_vec / norm
        
        model_type = self.embedding_model_type
        if model_type in self.command_vectors and self.command_vectors[model_type].size > 0:
            arr = self.command_vectors[model_type]  # N x D normalized
            sims = arr.dot(user_vec_norm)
            idx = int(np.argmax(sims))
            return self.commands_data[idx], float(sims[idx])
        else:
            # Fallback: compute pairwise
            max_sim = 0.0
            best = None
            for command in self.commands_data:
                cmd_vec = self.word_embedding.text_to_vector(command['text'])
                sim = self.calculate_similarity(user_vec_norm, cmd_vec)
                if sim > max_sim:
                    max_sim = sim
                    best = command
            return best, max_sim       
    
    def levenshtein_distance(self, s1, s2):
        """Calcula a distância de Levenshtein entre duas strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def correct_spelling(self, word):
        """Corrige erros ortográficos usando Levenshtein"""
        known_words = [
            'azul', 'vermelho', 'verde', 'amarelo', 'roxo', 'laranja', 
            'rosa', 'cinza', 'preto', 'branco', 'ciano', 'magenta',
            'esquerda', 'direita', 'cima', 'baixo', 'centro', 'topo', 'abaixo',
            'mude', 'mudar', 'coloque', 'colocar', 'aumente', 'aumentar',
            'diminua', 'diminuir', 'mova', 'mover', 'navegue', 'navegar',
            'fundo', 'fonte', 'texto', 'menu', 'página', 'pagina', 'layout'
        ]
        
        if word in known_words:
            return word
        
        min_distance = float('inf')
        best_match = word
        
        for known_word in known_words:
            distance = self.levenshtein_distance(word, known_word)
            if distance < min_distance and distance <= 2:
                min_distance = distance
                best_match = known_word
        
        if best_match != word:
            print(f"🔧 Correção ortográfica: '{word}' → '{best_match}'")
        
        return best_match
    
    def preprocess_with_correction(self, text):
        """Pré-processa texto com correção ortográfica"""
        words = self.word_embedding.preprocess_text(text)
        corrected_words = [self.correct_spelling(word) for word in words]
        return ' '.join(corrected_words)
    
    def extract_parameters(self, user_input, command_template):
        """Extrai parâmetros do comando do usuário"""
        params = command_template['params'].copy()
        
        color_map = {
            'azul': '#0066cc', 'vermelho': '#cc0000', 'verde': '#00cc00',
            'amarelo': '#ffcc00', 'roxo': '#9900cc', 'laranja': '#ff6600',
            'rosa': '#ff66cc', 'cinza': '#808080', 'preto': '#000000', 
            'branco': '#ffffff', 'ciano': '#00cccc', 'magenta': '#cc00cc',
            'blue': '#0066cc', 'red': '#cc0000', 'green': '#00cc00'
        }
        
        direction_map = {
            'esquerda': 'left', 'direita': 'right',
            'topo': 'top', 'cima': 'top', 'baixo': 'bottom', 'abaixo': 'bottom',
            'centro': 'center', 'meio': 'center'
        }
        
        page_map = {
            'inicial': 'home', 'home': 'home', 'principal': 'home',
            'contato': 'contact', 'contatos': 'contact',
            'sobre': 'about', 'sobre nós': 'about', 'sobre nos': 'about',
            'configurações': 'settings', 'configuracoes': 'settings', 'config': 'settings'
        }
        
        words = self.word_embedding.preprocess_text(user_input)
        for word in words:
            if word in color_map:
                params['color'] = color_map[word]
            elif word in direction_map:
                params['direction'] = direction_map[word]
            elif word in page_map:
                params['page'] = page_map[word]
        
        return params
    
    def extract_keywords(self, text):
        """Extrai palavras-chave importantes do texto"""
        words = self.word_embedding.preprocess_text(text)
        
        keywords = {
            'increase': ['aumente', 'aumentar', 'maior', 'cresça', 'amplie', 'eleve'],
            'decrease': ['diminua', 'diminuir', 'menor', 'reduza', 'abaixe'],
            'change': ['mude', 'mudar', 'altere', 'alterar', 'troque', 'trocar'],
            'move': ['mova', 'mover', 'coloque', 'colocar', 'desloque'],
            'navigate': ['vá', 'ir', 'navegue', 'navegar', 'acesse', 'acessar', 'abra'],
            'reset': ['redefinir', 'resetar', 'restaurar', 'voltar', 'padrão']
        }
        
        found = set()
        for word in words:
            for category, keyword_list in keywords.items():
                if word in keyword_list:
                    found.add(category)
        
        return found
    
    def validate_action_match(self, user_input, suggested_command):
        """Valida se a ação sugerida corresponde à intenção do usuário"""
        user_keywords = self.extract_keywords(user_input)
        suggested_keywords = self.extract_keywords(suggested_command['text'])
        
        # Verifica conflito de AÇÃO
        opposite_pairs = [
            {'increase', 'decrease'},
            {'left', 'right'},
            {'top', 'bottom'},
        ]
        
        for pair in opposite_pairs:
            user_has_one = len(user_keywords & pair) == 1
            suggested_has_opposite = len(suggested_keywords & pair) == 1
            
            if user_has_one and suggested_has_opposite:
                user_action = list(user_keywords & pair)[0]
                suggested_action = list(suggested_keywords & pair)[0]
                
                if user_action != suggested_action:
                    print(f"⚠️ CONFLITO DE AÇÃO: usuário quer '{user_action}' mas sugestão é '{suggested_action}'")
                    return False
        
        # Verifica conflito de PARÂMETROS
        user_params = self.extract_parameters(user_input, suggested_command)
        suggested_params = suggested_command['params']
        
        # Verifica conflito de COR (CORRIGIDO)
        if 'color' in user_params and 'color' in suggested_params:
            color_map_complete = {
                'azul': '#0066cc', 'vermelho': '#cc0000', 'verde': '#00cc00',
                'amarelo': '#ffcc00', 'roxo': '#9900cc', 'laranja': '#ff6600',
                'rosa': '#ff66cc', 'cinza': '#808080', 'preto': '#000000',
                'branco': '#ffffff', 'ciano': '#00cccc', 'magenta': '#cc00cc',
                'blue': '#0066cc', 'red': '#cc0000', 'green': '#00cc00',
                'yellow': '#ffcc00', 'purple': '#9900cc', 'orange': '#ff6600',
                'pink': '#ff66cc', 'gray': '#808080', 'grey': '#808080',
                'black': '#000000', 'white': '#ffffff', 'cyan': '#00cccc',
                '#0066cc': 'azul', '#cc0000': 'vermelho', '#00cc00': 'verde',
                '#ffcc00': 'amarelo', '#9900cc': 'roxo', '#ff6600': 'laranja',
                '#ff66cc': 'rosa', '#808080': 'cinza', '#000000': 'preto',
                '#ffffff': 'branco', '#00cccc': 'ciano', '#cc00cc': 'magenta'
            }
            
            user_color = user_params.get('color', '')
            suggested_color = suggested_params.get('color', '')
            
            def normalize_to_hex(color_value):
                if not color_value:
                    return ''
                if color_value.startswith('#'):
                    return color_value
                return color_map_complete.get(color_value, color_value)
            
            user_color_hex = normalize_to_hex(user_color)
            suggested_color_hex = normalize_to_hex(suggested_color)
            
            if user_color_hex and suggested_color_hex and user_color_hex != suggested_color_hex:
                user_color_name = color_map_complete.get(user_color_hex, user_color_hex)
                suggested_color_name = color_map_complete.get(suggested_color_hex, suggested_color_hex)
                print(f"⚠️ CONFLITO DE COR: usuário quer '{user_color_name}' mas sugestão é '{suggested_color_name}'")
                return False
        
        # Verifica conflito de DIREÇÃO
        if 'direction' in user_params and 'direction' in suggested_params:
            user_direction = user_params.get('direction', '')
            suggested_direction = suggested_params.get('direction', '')
            
            if user_direction and suggested_direction and user_direction != suggested_direction:
                print(f"⚠️ CONFLITO DE DIREÇÃO: usuário quer '{user_direction}' mas sugestão é '{suggested_direction}'")
                return False
        
        # Verifica conflito de PÁGINA
        if 'page' in user_params and 'page' in suggested_params:
            user_page = user_params.get('page', '')
            suggested_page = suggested_params.get('page', '')
            
            if user_page and suggested_page and user_page != suggested_page:
                print(f"⚠️ CONFLITO DE PÁGINA: usuário quer '{user_page}' mas sugestão é '{suggested_page}'")
                return False
        
        return True
    
    def process_command(self, user_input):
        """Processa comando usando sistema híbrido inteligente"""
        
        # Aplica correção ortográfica
        corrected_input = self.preprocess_with_correction(user_input)
        
        # Detecta negações
        normalized_input = user_input.lower().strip()
        negation_patterns = ['não quero', 'nao quero', 'não faça', 'nao faca', 
                            'não mude', 'nao mude', 'não vou', 'nao vou']
        
        if any(normalized_input.startswith(pattern) for pattern in negation_patterns):
            print(f"❌ NEGAÇÃO DETECTADA: usuário não quer executar ação")
            return {
                'action': 'negation_detected',
                'original_input': user_input,
                'message': 'Entendi, você NÃO quer fazer isso. Como posso ajudar então?'
            }
        
        # Verifica confirmação
        confirmations_yes = ['sim', 'yes', 'ok', 'isso', 'exato', 'correto', 'uhum', 's']
        confirmations_no = ['não', 'nao', 'no', 'negativo', 'errado', 'n']
        
        if self.waiting_confirmation and self.last_suggestion:
            if normalized_input in confirmations_yes:
                print(f"✅ CONFIRMAÇÃO POSITIVA - Executando sugestão")
                self.waiting_confirmation = False
                suggested_command = self.last_suggestion
                self.last_suggestion = None
                
                params = self.extract_parameters(suggested_command['text'], suggested_command)
                return {
                    'action': suggested_command['action'],
                    'params': params,
                    'confidence': 1.0,
                    'original_input': user_input,
                    'method': 'user_confirmation',
                    'confirmed_command': suggested_command['text']
                }
            
            elif normalized_input in confirmations_no:
                print(f"❌ CONFIRMAÇÃO NEGATIVA - Cancelando sugestão")
                self.waiting_confirmation = False
                self.last_suggestion = None
                
                return {
                    'action': 'cancel_suggestion',
                    'original_input': user_input,
                    'message': 'Ok, entendi. Por favor, reformule seu comando de forma mais específica ou detalhada.'
                }
        
        if self.waiting_confirmation:
            print(f"⚠️ Usuário não respondeu sim/não, processando novo comando")
            self.waiting_confirmation = False
            self.last_suggestion = None
        
        # Processamento normal - garante vetor válido
        user_vector = self.word_embedding.text_to_vector(corrected_input)
        if user_vector is None:
            # Garante um vetor de zeros com dimensão do embedding atual
            user_vector = np.zeros(getattr(self.word_embedding, 'vector_size', 0))
        if isinstance(user_vector, np.ndarray):
            user_vector = user_vector.tolist()
        
        # Primeiro tenta MLP se estiver pronto; captura exceptions (dimensão, etc.)
        predicted_class = None
        confidence = 0.0
        if self.mlp is not None:
            try:
                predicted_class, confidence = self.mlp.predict_class(user_vector)
            except Exception as e:
                print(f"⚠️ Falha no MLP predict (provável mismatch de dimensão): {e}")
                predicted_class, confidence = None, 0.0
        
        action = self.label_decoder.get(predicted_class) if predicted_class is not None else None
        
        # Similaridade (sempre disponível) - busca mais similar
        similar_command, similarity = self.find_most_similar_command(corrected_input)
        
        print(f"🔍 Input: '{user_input}'")
        if corrected_input != user_input.lower():
            print(f"🔧 Corrigido: '{corrected_input}'")
        if action:
            print(f"🤖 MLP: {action} ({confidence:.2%})")
        else:
            print(f"🤖 MLP: indisponível ou sem predição confiável")
        
        if similar_command:
            print(f"📊 Similar: '{similar_command['text']}' ({similarity:.2%})")
        else:
            print(f"📊 Similar: Nenhum comando similar encontrado")
        
        # Sistema de decisão híbrido - ordem das verificações:
        # 1) Similaridade alta -> executa direto
        if similar_command and similarity >= self.high_similarity_threshold:
            print(f"✅ EXECUÇÃO DIRETA POR SIMILARIDADE ({similarity:.2%})")
            params = self.extract_parameters(corrected_input, similar_command)
            return {
                'action': similar_command['action'],
                'params': params,
                'confidence': similarity,
                'original_input': user_input,
                'method': 'similarity_direct',
                'similar_command': similar_command['text']
            }
        
        # 2) MLP disponível e confiante -> executa por MLP
        if predicted_class is not None and confidence >= self.confidence_threshold:
            print(f"✅ EXECUÇÃO POR MLP ({confidence:.2%})")
            for cmd in self.commands_data:
                if cmd['action'] == action:
                    params = self.extract_parameters(corrected_input, cmd)
                    return {
                        'action': action,
                        'params': params,
                        'confidence': confidence,
                        'original_input': user_input,
                        'method': 'mlp'
                    }
        
        # 3) Similaridade razoável -> sugere
        if similar_command and similarity >= self.similarity_threshold:
            # garantia extra: validação semântica (pode usar validate_action_match)
            if self.validate_action_match(user_input, similar_command):
                print(f"💡 SUGESTÃO ({similarity:.2%}) - Aguardando confirmação")
                self.last_suggestion = similar_command
                self.waiting_confirmation = True
                return {
                    'action': 'suggest',
                    'original_input': user_input,
                    'suggested_command': similar_command['text'],
                    'similarity': similarity,
                    'confidence': confidence,
                    'waiting_confirmation': True
                }
            else:
                print("⚠️ Sugestão rejeitada por validação de parâmetros/ação; prosseguindo para fallback")
        
        # 4) fallback: desconhecido - tenta retornar sugestão informal baseada em similaridade, senão desconhecido
        print(f"❌ NÃO RECONHECIDO")
        words = self.word_embedding.preprocess_text(corrected_input)
        if not words or len(words) == 0:
            message = "Comando vazio ou não reconhecido. Por favor, seja mais específico."
        elif len(words) == 1 and len(user_input) < 5:
            message = f"Comando muito curto: '{user_input}'. Por favor, descreva melhor o que deseja fazer."
        else:
            # Se existia similaridade baixa, ainda pode retornar sugestão fraca
            if similar_command and similarity > 0.2:
                message = f"Não reconheci com confiança. Talvez você quis dizer: '{similar_command['text']}' (similaridade {similarity:.2%})."
            else:
                message = "Comando não reconhecido. Por favor, reformule seu comando."
        
        return {
            'action': 'unknown',
            'original_input': user_input,
            'message': message,
            'confidence': confidence,
            'similarity': similarity if similar_command else 0.0
        }
    def execute_action(self, action_data):
        """Executa a ação e retorna mensagem"""
        action = action_data['action']
        params = action_data.get('params', {})
        method = action_data.get('method', '')
        
        method_msg = ""
        if method in ['similarity_direct', 'similarity']:
            method_msg = f" (interpretado como: '{action_data.get('similar_command', '')}')"
        elif method == 'user_confirmation':
            method_msg = f" (você confirmou: '{action_data.get('confirmed_command', '')}')"
        
        if action == 'change_background_color':
            color = params.get('color', '#0066cc')
            self.current_state['background_color'] = color
            return f"✅ Cor de fundo alterada para {color}{method_msg}"
        
        elif action == 'change_text_color':
            color = params.get('color', '#000000')
            self.current_state['text_color'] = color
            return f"✅ Cor do texto alterada para {color}{method_msg}"
        
        elif action == 'increase_font_size':
            self.current_state['font_size'] += 2
            return f"✅ Tamanho da fonte aumentado para {self.current_state['font_size']}px{method_msg}"
        
        elif action == 'decrease_font_size':
            self.current_state['font_size'] = max(10, self.current_state['font_size'] - 2)
            return f"✅ Tamanho da fonte diminuído para {self.current_state['font_size']}px{method_msg}"
        
        elif action == 'move_component':
            component = params.get('component', 'menu')
            direction = params.get('direction', 'left')
            
            if component == 'menu':
                self.current_state['menu_position'] = direction
                return f"✅ Menu movido para {direction}{method_msg}"
            elif component in ['header', 'cabeçalho', 'cabecalho']:
                self.current_state['header_position'] = direction
                return f"✅ Cabeçalho movido para {direction}{method_msg}"
            
            return f"✅ {component.capitalize()} movido para {direction}{method_msg}"
        
        elif action == 'navigate':
            page = params.get('page', 'home')
            self.current_state['current_page'] = page
            return f"✅ Navegando para página {page}{method_msg}"
        
        elif action == 'reset_layout':
            self.current_state = {
                'background_color': '#ffffff',
                'text_color': '#000000',
                'font_size': 16,
                'menu_position': 'center',
                'current_page': 'home',
                'header_position': 'top'
            }
            return "✅ Layout redefinido para configurações padrão"
        
        elif action == 'suggest':
            waiting = action_data.get('waiting_confirmation', False)
            base_msg = f"Você quis dizer: '{action_data['suggested_command']}'"
            
            if waiting:
                return f"💡 Não tenho certeza sobre o comando. {base_msg}? (Digite 'sim' para confirmar ou 'não' para reformular)"
            else:
                return f"💡 Não tenho certeza sobre o comando. {base_msg}?"
        
        elif action == 'cancel_suggestion':
            return f"❌ {action_data.get('message', 'Sugestão cancelada.')}"
        
        elif action == 'negation_detected':
            return f"❌ {action_data.get('message', 'Entendi que você não quer fazer isso.')}"
        
        elif action == 'unknown':
            confidence_pct = action_data.get('confidence', 0) * 100
            similarity_pct = action_data.get('similarity', 0) * 100
            return f"❌ {action_data.get('message', 'Comando não reconhecido.')} (confiança: {confidence_pct:.0f}%, similaridade: {similarity_pct:.0f}%)"
        
        else:
            return f"⚠️ Ação '{action}' não implementada."
    
    def process_user_input(self, user_input):
        """Processa entrada do usuário e retorna resposta completa"""
        print(f"\n{'='*60}")
        
        action_data = self.process_command(user_input)
        result_message = self.execute_action(action_data)
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        frontend_update = {
            'backgroundColor': self.current_state['background_color'],
            'textColor': self.current_state['text_color'],
            'fontSize': f"{self.current_state['font_size']}px",
            'menuPosition': self.current_state['menu_position'],
            'currentPage': self.current_state['current_page'],
            'headerPosition': self.current_state['header_position']
        }
        
        response = {
            'user_input': user_input,
            'assistant_response': result_message,
            'action_data': convert_numpy(action_data),
            'frontend_update': frontend_update
        }
        
        self.chat_history.append(response)
        print(f"{'='*60}\n")
        return response
