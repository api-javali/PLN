"""
Assistente Virtual COMPLETO E CORRIGIDO
Usa Word2Vec (Gensim) + MLP + Sistema Híbrido

Correções aplicadas:
1. ✅ Sistema de confirmação (sim/não)
2. ✅ Validação de conflitos de ação (aumentar vs diminuir)
3. ✅ Validação de conflitos de parâmetros (cores, direções, páginas)
4. ✅ Proteção contra NoneType
5. ✅ Estrutura de código limpa e organizada
"""

import json
import numpy as np
from assistente.word2vec import Word2VecEmbedding
from assistente.classifier import MLPClassifier


class VirtualAssistant:
    def __init__(self):
        # Word2Vec profissional (Gensim)
        self.word_embedding = Word2VecEmbedding(
            vector_size=50,  # 50 dimensões (bom compromisso)
            window=3,        # Janela de contexto de 3 palavras
            epochs=100       # 100 épocas de treinamento
        )
        
        self.mlp = None
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
        
        # 🆕 Sistema de confirmação
        self.last_suggestion = None
        self.waiting_confirmation = False
        
        # Thresholds configuráveis
        self.confidence_threshold = 0.60  # 60% para MLP
        self.similarity_threshold = 0.45   # 45% para sugestão
        self.high_similarity_threshold = 0.60  # 60% para execução automática
        
        # Inicializa
        self.load_training_data()
        self.train_models()
    
    def load_training_data(self):
        """Carrega dados de treinamento do arquivo JSON"""
        try:
            with open('training_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.commands_data = data.get('commands', [])
                print(f"✅ {len(self.commands_data)} comandos carregados do training_data.json")
        except FileNotFoundError:
            print("⚠️ Arquivo training_data.json não encontrado")
            raise
    
    def train_models(self):
        """Treina Word2Vec e MLP"""
        print("\n" + "="*60)
        print("🧠 TREINAMENTO DO ASSISTENTE VIRTUAL")
        print("="*60)
        
        # Prepara dados
        texts = [cmd['text'] for cmd in self.commands_data]
        actions = sorted(set(cmd['action'] for cmd in self.commands_data))
        
        # Cria encoders de labels
        self.label_encoder = {action: idx for idx, action in enumerate(actions)}
        self.label_decoder = {idx: action for action, idx in self.label_encoder.items()}
        
        # 1. Treina Word2Vec
        print("\n📝 ETAPA 1: Word2Vec (Gensim)")
        tokenized_texts = []
        for text in texts:
            words = self.word_embedding.preprocess_text(text)
            tokenized_texts.append(words)
        
        self.word_embedding.build_vocab(tokenized_texts)
        
        # 2. Prepara dados para o MLP
        print("\n🔢 ETAPA 2: Preparando dados para MLP")
        X = []
        y = []
        for cmd in self.commands_data:
            # Converte texto para vetor usando Word2Vec
            vector = self.word_embedding.text_to_vector(cmd['text'])
            X.append(vector.tolist())  # Converte numpy array para lista
            
            # One-hot encoding para o label
            label_vector = [0.0] * len(actions)
            label_vector[self.label_encoder[cmd['action']]] = 1.0
            y.append(label_vector)
        
        print(f"   • Exemplos de treinamento: {len(X)}")
        print(f"   • Dimensão dos vetores: {len(X[0])}")
        print(f"   • Classes: {len(actions)}")
        
        # 3. Treina MLP
        print("\n🤖 ETAPA 3: MLP com Backpropagation")
        input_size = self.word_embedding.vector_size
        hidden_size = min(30, max(10, input_size // 2))
        output_size = len(actions)
        
        print(f"   • Arquitetura: {input_size} → {hidden_size} → {output_size}")
        print(f"   • Learning rate: 0.15")
        print(f"   • Épocas: 200")
        
        self.mlp = MLPClassifier(input_size, hidden_size, output_size, learning_rate=0.15)
        self.mlp.train(X, y, epochs=200)
        
        print("\n" + "="*60)
        print("✅ TREINAMENTO CONCLUÍDO")
        print("="*60)
        print(f"📊 Resumo:")
        print(f"   • Vocabulário: {len(self.word_embedding.model.wv)} palavras")
        print(f"   • Ações: {', '.join(actions)}")
        print(f"   • Threshold confiança: {self.confidence_threshold*100}%")
        print(f"   • Threshold similaridade: {self.similarity_threshold*100}%")
        print("="*60 + "\n")
    
    def calculate_similarity(self, vec1, vec2):
        """Calcula similaridade cosseno entre dois vetores"""
        return self.word_embedding.cosine_similarity(vec1, vec2)
    
    def find_most_similar_command(self, user_input):
        """Encontra o comando mais similar no treinamento"""
        user_vector = self.word_embedding.text_to_vector(user_input)
        max_similarity = 0.0
        most_similar = None
        
        # 🔧 PROTEÇÃO: Se não há palavras válidas, retorna None
        if not any(user_vector):  # Vetor vazio ou só zeros
            return None, 0.0
        
        for command in self.commands_data:
            command_vector = self.word_embedding.text_to_vector(command['text'])
            similarity = self.calculate_similarity(user_vector, command_vector)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = command
        
        return most_similar, max_similarity
    
    def extract_parameters(self, user_input, command_template):
        """Extrai parâmetros do comando do usuário"""
        params = command_template['params'].copy()
        
        # Mapeamento de cores
        color_map = {
            'azul': '#0066cc', 'vermelho': '#cc0000', 'verde': '#00cc00',
            'amarelo': '#ffcc00', 'roxo': '#9900cc', 'laranja': '#ff6600',
            'rosa': '#ff66cc', 'cinza': '#808080', 'preto': '#000000', 
            'branco': '#ffffff', 'ciano': '#00cccc', 'magenta': '#cc00cc',
            'blue': '#0066cc', 'red': '#cc0000', 'green': '#00cc00'
        }
        
        # Mapeamento de direções
        direction_map = {
            'esquerda': 'left', 'direita': 'right',
            'topo': 'top', 'cima': 'top', 'baixo': 'bottom', 'abaixo': 'bottom',
            'centro': 'center', 'meio': 'center'
        }
        
        # Mapeamento de páginas
        page_map = {
            'inicial': 'home', 'home': 'home', 'principal': 'home',
            'contato': 'contact', 'contatos': 'contact',
            'sobre': 'about', 'sobre nós': 'about', 'sobre nos': 'about',
            'configurações': 'settings', 'configuracoes': 'settings', 'config': 'settings'
        }
        
        # Extrai parâmetros baseados nas palavras do usuário
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
        """
        Extrai palavras-chave importantes do texto
        Para validar se comando corresponde à ação
        """
        words = self.word_embedding.preprocess_text(text)
        
        # Dicionário de palavras-chave por categoria
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
        """
        Valida se a ação sugerida corresponde à intenção do usuário
        ✅ Valida AÇÕES e PARÂMETROS (cores, direções, páginas)
        
        Returns:
            bool: True se ação é válida, False se conflita
        """
        user_keywords = self.extract_keywords(user_input)
        suggested_keywords = self.extract_keywords(suggested_command['text'])
        
        # 1. Verifica conflito de AÇÃO (aumentar vs diminuir)
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
        
        # 2. Verifica conflito de PARÂMETROS
        user_params = self.extract_parameters(user_input, suggested_command)
        suggested_params = suggested_command['params']
        
        # 2.1 Verifica conflito de COR
        if 'color' in user_params and 'color' in suggested_params:
            color_names = {
                '#0066cc': 'azul', '#cc0000': 'vermelho', '#00cc00': 'verde',
                '#ffcc00': 'amarelo', '#9900cc': 'roxo', '#ff6600': 'laranja',
                '#ff66cc': 'rosa', '#808080': 'cinza', '#000000': 'preto',
                '#ffffff': 'branco', '#00cccc': 'ciano', '#cc00cc': 'magenta'
            }
            
            user_color = user_params.get('color', '')
            suggested_color = suggested_params.get('color', '')
            
            if user_color and suggested_color and user_color != suggested_color:
                user_color_name = color_names.get(user_color, user_color)
                suggested_color_name = color_names.get(suggested_color, suggested_color)
                print(f"⚠️ CONFLITO DE COR: usuário quer '{user_color_name}' mas sugestão é '{suggested_color_name}'")
                return False
        
        # 2.2 Verifica conflito de DIREÇÃO
        if 'direction' in user_params and 'direction' in suggested_params:
            user_direction = user_params.get('direction', '')
            suggested_direction = suggested_params.get('direction', '')
            
            if user_direction and suggested_direction and user_direction != suggested_direction:
                print(f"⚠️ CONFLITO DE DIREÇÃO: usuário quer '{user_direction}' mas sugestão é '{suggested_direction}'")
                return False
        
        # 2.3 Verifica conflito de PÁGINA
        if 'page' in user_params and 'page' in suggested_params:
            user_page = user_params.get('page', '')
            suggested_page = suggested_params.get('page', '')
            
            if user_page and suggested_page and user_page != suggested_page:
                print(f"⚠️ CONFLITO DE PÁGINA: usuário quer '{user_page}' mas sugestão é '{suggested_page}'")
                return False
        
        return True
    
    def process_command(self, user_input):
        """
        Processa comando usando sistema híbrido inteligente:
        1. Word2Vec converte texto para vetor
        2. MLP classifica a ação
        3. Sistema de confirmação (sim/não)
        4. Validação de conflitos
        5. Sugere comando similar ou executa
        """
        
        # ============================================
        # STEP 1: Verifica se é uma resposta de confirmação
        # ============================================
        normalized_input = user_input.lower().strip()
        
        # Lista de confirmações positivas
        confirmations_yes = ['sim', 'yes', 'ok', 'isso', 'exato', 'correto', 'uhum', 's']
        # Lista de confirmações negativas
        confirmations_no = ['não', 'nao', 'no', 'negativo', 'errado', 'n']
        
        # Se está aguardando confirmação e usuário respondeu
        if self.waiting_confirmation and self.last_suggestion:
            
            # Resposta positiva: EXECUTA a sugestão
            if normalized_input in confirmations_yes:
                print(f"✅ CONFIRMAÇÃO POSITIVA - Executando sugestão")
                
                # Reseta flags
                self.waiting_confirmation = False
                suggested_command = self.last_suggestion
                self.last_suggestion = None
                
                # Extrai parâmetros do comando sugerido (não do "sim")
                params = self.extract_parameters(suggested_command['text'], suggested_command)
                return {
                    'action': suggested_command['action'],
                    'params': params,
                    'confidence': 1.0,  # 100% pois foi confirmado
                    'original_input': user_input,
                    'method': 'user_confirmation',
                    'confirmed_command': suggested_command['text']
                }
            
            # Resposta negativa: CANCELA e pede reformulação
            elif normalized_input in confirmations_no:
                print(f"❌ CONFIRMAÇÃO NEGATIVA - Cancelando sugestão")
                
                # Reseta flags
                self.waiting_confirmation = False
                self.last_suggestion = None
                
                return {
                    'action': 'cancel_suggestion',
                    'original_input': user_input,
                    'message': 'Ok, entendi. Por favor, reformule seu comando de forma mais específica ou detalhada.'
                }
        
        # Se usuário não respondeu sim/não quando esperado, reseta contexto
        if self.waiting_confirmation:
            print(f"⚠️ Usuário não respondeu sim/não, processando novo comando")
            self.waiting_confirmation = False
            self.last_suggestion = None
        
        # ============================================
        # STEP 2: Processamento normal do comando
        # ============================================
        user_vector = self.word_embedding.text_to_vector(user_input)
        
        if isinstance(user_vector, np.ndarray):
            user_vector = user_vector.tolist()
        
        predicted_class, confidence = self.mlp.predict_class(user_vector)
        action = self.label_decoder[predicted_class]
        
        similar_command, similarity = self.find_most_similar_command(user_input)
        
        print(f"🔍 Input: '{user_input}'")
        print(f"🤖 MLP: {action} ({confidence:.2%})")
        
        if similar_command:
            print(f"📊 Similar: '{similar_command['text']}' ({similarity:.2%})")
        else:
            print(f"📊 Similar: Nenhum comando similar encontrado")
        
        # ============================================
        # Sistema de decisão híbrido com 4 níveis
        # ============================================
        
        # Nível 1: Similaridade MUITO alta (>60%) → Valida e Executa
        if similar_command and similarity >= self.high_similarity_threshold:
            
            # Validação: Verifica se ação corresponde à intenção
            if self.validate_action_match(user_input, similar_command):
                print(f"✅ EXECUÇÃO POR SIMILARIDADE ({similarity:.2%})")
                params = self.extract_parameters(user_input, similar_command)
                return {
                    'action': similar_command['action'],
                    'params': params,
                    'confidence': similarity,
                    'original_input': user_input,
                    'method': 'similarity',
                    'similar_command': similar_command['text']
                }
            else:
                # Se há conflito, SUGERE ao invés de executar
                print(f"💡 SIMILARIDADE ALTA MAS COM CONFLITO - Sugerindo")
                
                self.last_suggestion = similar_command
                self.waiting_confirmation = True
                
                return {
                    'action': 'suggest',
                    'original_input': user_input,
                    'suggested_command': similar_command['text'],
                    'similarity': similarity,
                    'confidence': confidence,
                    'waiting_confirmation': True,
                    'conflict_detected': True
                }
        
        # Nível 2: Confiança MLP boa (>60%) → Usa MLP
        elif confidence >= self.confidence_threshold:
            print(f"✅ EXECUÇÃO POR MLP ({confidence:.2%})")
            for cmd in self.commands_data:
                if cmd['action'] == action:
                    params = self.extract_parameters(user_input, cmd)
                    return {
                        'action': action,
                        'params': params,
                        'confidence': confidence,
                        'original_input': user_input,
                        'method': 'mlp'
                    }
        
        # Nível 3: Similaridade razoável (>45%) → SUGERE e AGUARDA
        elif similar_command and similarity >= self.similarity_threshold:
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
        
        # Nível 4: Não reconheceu
        else:
            print(f"❌ NÃO RECONHECIDO")
            
            words = self.word_embedding.preprocess_text(user_input)
            if not words or len(words) == 0:
                message = "Comando vazio ou não reconhecido. Por favor, seja mais específico."
            elif len(words) == 1 and len(user_input) < 5:
                message = f"Comando muito curto: '{user_input}'. Por favor, descreva melhor o que deseja fazer."
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
        
        # Mensagem adicional se usou similaridade ou confirmação
        method_msg = ""
        if method == 'similarity':
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
            conflict = action_data.get('conflict_detected', False)
            
            base_msg = f"Você quis dizer: '{action_data['suggested_command']}'"
            
            if conflict:
                return f"⚠️ Detectei um possível conflito. {base_msg}? (Digite 'sim' para confirmar ou 'não' para reformular)"
            elif waiting:
                return f"💡 Não tenho certeza sobre o comando. {base_msg}? (Digite 'sim' para confirmar ou 'não' para reformular)"
            else:
                return f"💡 Não tenho certeza sobre o comando. {base_msg}?"
        
        elif action == 'cancel_suggestion':
            return f"❌ {action_data.get('message', 'Sugestão cancelada.')}"
        
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
        
        # Converte numpy para tipos nativos Python
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
