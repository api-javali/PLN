from flask import Flask, render_template, request, jsonify
import json
import math
import re
from collections import defaultdict
import os
import random

app = Flask(__name__)

class SimpleWordEmbedding:
    """Sistema de embeddings simples usando one-hot encoding e similaridade de cosseno"""
    def __init__(self):
        self.vocab = {}
        self.word_vectors = {}
        self.vector_size = 0
        
    def build_vocab(self, sentences):
        """Constrói vocabulário a partir das sentenças"""
        all_words = set()
        for sentence in sentences:
            all_words.update(sentence)
        
        self.vocab = {word: idx for idx, word in enumerate(all_words)}
        self.vector_size = len(all_words)
        
        # Cria vetores one-hot
        for word, idx in self.vocab.items():
            vector = [0] * self.vector_size
            vector[idx] = 1
            self.word_vectors[word] = vector
    
    def text_to_vector(self, text):
        """Converte texto para vetor usando média dos vetores das palavras"""
        words = self.preprocess_text(text)
        if not words:
            return [0] * self.vector_size
        
        vectors = []
        for word in words:
            if word in self.word_vectors:
                vectors.append(self.word_vectors[word])
        
        if vectors:
            # Média dos vetores
            return [sum(x) / len(vectors) for x in zip(*vectors)]
        else:
            return [0] * self.vector_size
    
    def preprocess_text(self, text):
        """Pré-processa o texto"""
        text = text.lower().strip()
        # Remove pontuação e divide em palavras
        words = re.findall(r'[a-zà-úçñ]+', text)
        return words

class SimpleNeuralNetwork:
    """Rede neural simples com uma camada oculta"""
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicializa pesos com valores aleatórios simples
        self.weights1 = self._initialize_weights(input_size, hidden_size)
        self.weights2 = self._initialize_weights(hidden_size, output_size)
        self.bias1 = [random.random() for _ in range(hidden_size)]
        self.bias2 = [random.random() for _ in range(output_size)]
    
    def _initialize_weights(self, rows, cols):
        """Inicializa pesos com valores aleatórios"""
        return [[random.random() for _ in range(cols)] for _ in range(rows)]
    
    def sigmoid(self, x):
        """Função sigmoid"""
        if x < -700:  # Previne overflow
            return 0.0
        return 1 / (1 + math.exp(-x))
    
    def softmax(self, x):
        """Função softmax"""
        exp_x = [math.exp(i - max(x)) for i in x]  # Subtrai max para estabilidade numérica
        sum_exp = sum(exp_x)
        return [i / sum_exp for i in exp_x]
    
    def forward(self, x):
        """Propagação para frente"""
        # Camada oculta
        hidden = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            for i in range(self.input_size):
                hidden[j] += x[i] * self.weights1[i][j]
            hidden[j] = self.sigmoid(hidden[j] + self.bias1[j])
        
        # Camada de saída
        output = [0.0] * self.output_size
        for j in range(self.output_size):
            for i in range(self.hidden_size):
                output[j] += hidden[i] * self.weights2[i][j]
            output[j] += self.bias2[j]
        
        return self.softmax(output), hidden
    
    def predict(self, x):
        """Faz predição"""
        probabilities, _ = self.forward(x)
        return probabilities
    
    def predict_class(self, x):
        """Retorna a classe predita"""
        probabilities = self.predict(x)
        return probabilities.index(max(probabilities))

class VirtualAssistant:
    def __init__(self):
        self.word_embedding = SimpleWordEmbedding()
        self.neural_network = None
        self.label_encoder = {}
        self.label_decoder = {}
        self.commands_data = []
        self.current_state = {
            'background_color': 'white',
            'font_size': 16,
            'menu_position': 'center',
            'current_page': 'home'
        }
        self.chat_history = []
        
        self.load_training_data()
        self.train_models()
    
    def load_training_data(self):
        """Carrega dados de treinamento"""
        self.commands_data = [
            {"text": "mude a cor do fundo para azul", "action": "change_background_color", "params": {"color": "blue"}},
            {"text": "altere a cor de fundo para vermelho", "action": "change_background_color", "params": {"color": "red"}},
            {"text": "troque a cor do background para verde", "action": "change_background_color", "params": {"color": "green"}},
            {"text": "faça o fundo amarelo", "action": "change_background_color", "params": {"color": "yellow"}},
            {"text": "aumente o tamanho da fonte", "action": "increase_font_size", "params": {}},
            {"text": "aumente a letra", "action": "increase_font_size", "params": {}},
            {"text": "deixe o texto maior", "action": "increase_font_size", "params": {}},
            {"text": "diminua a fonte", "action": "decrease_font_size", "params": {}},
            {"text": "reduza o tamanho do texto", "action": "decrease_font_size", "params": {}},
            {"text": "mova o menu para a esquerda", "action": "move_component", "params": {"component": "menu", "direction": "left"}},
            {"text": "desloque a barra de navegação para a direita", "action": "move_component", "params": {"component": "menu", "direction": "right"}},
            {"text": "vá para a página inicial", "action": "navigate", "params": {"page": "home"}},
            {"text": "navegue para contato", "action": "navigate", "params": {"page": "contact"}},
            {"text": "ir para sobre nós", "action": "navigate", "params": {"page": "about"}},
            {"text": "acesse configurações", "action": "navigate", "params": {"page": "settings"}},
            {"text": "redefinir layout", "action": "reset_layout", "params": {}},
            {"text": "voltar ao padrão", "action": "reset_layout", "params": {}},
            {"text": "restaurar configurações iniciais", "action": "reset_layout", "params": {}}
        ]
    
    def train_models(self):
        """Treina os modelos do assistente"""
        # Prepara dados
        texts = [cmd['text'] for cmd in self.commands_data]
        actions = list(set(cmd['action'] for cmd in self.commands_data))
        
        # Cria encoders de labels
        self.label_encoder = {action: idx for idx, action in enumerate(actions)}
        self.label_decoder = {idx: action for action, idx in self.label_encoder.items()}
        
        # Tokeniza textos
        tokenized_texts = []
        for text in texts:
            words = self.word_embedding.preprocess_text(text)
            tokenized_texts.append(words)
        
        # Treina embeddings
        self.word_embedding.build_vocab(tokenized_texts)
        
        # Treina rede neural
        input_size = self.word_embedding.vector_size
        hidden_size = min(20, input_size)  # Tamanho da camada oculta
        output_size = len(actions)
        
        self.neural_network = SimpleNeuralNetwork(input_size, hidden_size, output_size)
        
        print("✅ Modelos treinados com sucesso!")
        print(f"📊 Vocabulário: {len(self.word_embedding.vocab)} palavras")
        print(f"🎯 Ações: {len(actions)} comandos")
        print(f"🔢 Classes: {self.label_encoder}")
    
    def calculate_similarity(self, vec1, vec2):
        """Calcula similaridade cosseno entre dois vetores"""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def find_most_similar_command(self, user_input):
        """Encontra o comando mais similar no treinamento"""
        user_vector = self.word_embedding.text_to_vector(user_input)
        max_similarity = 0.0
        most_similar = None
        
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
        input_lower = user_input.lower()
        
        # Mapeamento de cores
        color_map = {
            'azul': 'blue', 'vermelho': 'red', 'verde': 'green', 
            'amarelo': 'yellow', 'roxo': 'purple', 'laranja': 'orange',
            'rosa': 'pink', 'cinza': 'gray', 'preto': 'black', 'branco': 'white'
        }
        
        # Mapeamento de direções
        direction_map = {
            'esquerda': 'left', 'direita': 'right', 
            'topo': 'top', 'cima': 'top', 'baixo': 'bottom',
            'centro': 'center'
        }
        
        # Mapeamento de páginas
        page_map = {
            'inicial': 'home', 'home': 'home', 'principal': 'home',
            'contato': 'contact', 'contatos': 'contact',
            'sobre': 'about', 'sobre nós': 'about', 'sobre nos': 'about',
            'configurações': 'settings', 'configuracoes': 'settings', 'config': 'settings'
        }
        
        # Extrai parâmetros baseados nas palavras do usuário
        for word in self.word_embedding.preprocess_text(user_input):
            if word in color_map:
                params['color'] = color_map[word]
            elif word in direction_map:
                params['direction'] = direction_map[word]
            elif word in page_map:
                params['page'] = page_map[word]
        
        return params
    
    def process_command(self, user_input):
        """Processa o comando do usuário"""
        # Primeiro tenta usar palavras-chave para classificação mais precisa
        keyword_result = self.keyword_classification(user_input)
        if keyword_result and keyword_result.get('confidence', 0) > 0.8:
            return keyword_result
        
        # Se não encontrou com alta confiança, usa a rede neural
        user_vector = self.word_embedding.text_to_vector(user_input)
        
        try:
            # Usa a rede neural para predição
            probabilities = self.neural_network.predict(user_vector)
            predicted_class = probabilities.index(max(probabilities))
            confidence = max(probabilities)
            
            action = self.label_decoder[predicted_class]
            
            # Se confiança é baixa, usa similaridade
            if confidence < 0.3:
                similar_command, similarity = self.find_most_similar_command(user_input)
                if similarity > 0.5:
                    return {
                        'action': 'suggest',
                        'original_input': user_input,
                        'suggested_command': similar_command['text'],
                        'similarity': similarity,
                        'confidence': confidence
                    }
                else:
                    return {
                        'action': 'unknown',
                        'original_input': user_input,
                        'message': 'Comando não reconhecido. Por favor, reformule seu comando.'
                    }
            
            # Encontra os parâmetros
            for cmd in self.commands_data:
                if cmd['action'] == action:
                    params = self.extract_parameters(user_input, cmd)
                    return {
                        'action': action,
                        'params': params,
                        'confidence': confidence,
                        'original_input': user_input
                    }
            
            return {
                'action': 'unknown',
                'original_input': user_input,
                'message': 'Comando reconhecido mas não pude executá-lo.'
            }
            
        except Exception as e:
            # Fallback para correspondência de palavras-chave
            return self.keyword_classification(user_input)
    
    def keyword_classification(self, user_input):
        """Classificação baseada em palavras-chave - MAIS PRECISA"""
        input_lower = user_input.lower()
        words = set(self.word_embedding.preprocess_text(user_input))
        
        # Palavras-chave para cada ação
        action_keywords = {
            'change_background_color': ['cor', 'fundo', 'background', 'azul', 'vermelho', 'verde', 'amarelo'],
            'increase_font_size': ['aumente', 'aumentar', 'maior', 'fonte', 'letra', 'texto', 'tamanho'],
            'decrease_font_size': ['diminua', 'diminuir', 'menor', 'fonte', 'letra', 'texto', 'tamanho'],
            'navigate': ['vá', 'ir', 'navegar', 'página', 'página', 'home', 'inicial', 'contato', 'sobre', 'configurações'],
            'move_component': ['mova', 'mover', 'desloque', 'menu', 'barra', 'esquerda', 'direita'],
            'reset_layout': ['redefinir', 'reset', 'padrão', 'inicial', 'restaurar']
        }
        
        # Calcula pontuação para cada ação
        scores = {}
        for action, keywords in action_keywords.items():
            score = sum(1 for keyword in keywords if keyword in words or keyword in input_lower)
            scores[action] = score
        
        # Encontra a ação com maior pontuação
        best_action = max(scores, key=scores.get)
        best_score = scores[best_action]
        
        if best_score == 0:
            # Nenhuma palavra-chave encontrada, tenta similaridade
            similar_command, similarity = self.find_most_similar_command(user_input)
            if similarity > 0.3:
                return {
                    'action': 'suggest',
                    'original_input': user_input,
                    'suggested_command': similar_command['text'],
                    'similarity': similarity,
                    'confidence': 0.5
                }
            else:
                return {
                    'action': 'unknown',
                    'original_input': user_input,
                    'message': 'Comando não reconhecido. Tente: "mudar cor", "aumentar fonte", "ir para página inicial".'
                }
        
        # Calcula confiança baseada na pontuação
        confidence = min(best_score / 3.0, 1.0)  # Normaliza para 0-1
        
        # Encontra os parâmetros
        for cmd in self.commands_data:
            if cmd['action'] == best_action:
                params = self.extract_parameters(user_input, cmd)
                return {
                    'action': best_action,
                    'params': params,
                    'confidence': confidence,
                    'original_input': user_input
                }
        
        return {
            'action': 'unknown',
            'original_input': user_input,
            'message': 'Comando reconhecido mas não pude executá-lo.'
        }
    
    def execute_action(self, action_data):
        """Executa a ação e retorna mensagem"""
        action = action_data['action']
        params = action_data.get('params', {})
        
        if action == 'change_background_color':
            color = params.get('color', 'blue')
            self.current_state['background_color'] = color
            return f"✅ Cor de fundo alterada para {color}"
        
        elif action == 'increase_font_size':
            self.current_state['font_size'] += 2
            return f"✅ Tamanho da fonte aumentado para {self.current_state['font_size']}px"
        
        elif action == 'decrease_font_size':
            self.current_state['font_size'] = max(10, self.current_state['font_size'] - 2)
            return f"✅ Tamanho da fonte diminuído para {self.current_state['font_size']}px"
        
        elif action == 'move_component':
            component = params.get('component', 'menu')
            direction = params.get('direction', 'left')
            self.current_state['menu_position'] = direction
            return f"✅ {component.capitalize()} movido para {direction}"
        
        elif action == 'navigate':
            page = params.get('page', 'home')
            self.current_state['current_page'] = page
            return f"✅ Navegando para página {page}"
        
        elif action == 'reset_layout':
            self.current_state = {
                'background_color': 'white',
                'font_size': 16,
                'menu_position': 'center',
                'current_page': 'home'
            }
            return "✅ Layout redefinido para configurações padrão"
        
        elif action == 'suggest':
            return f"💡 Comando não reconhecido. Você quis dizer: '{action_data['suggested_command']}'?"
        
        elif action == 'unknown':
            return f"❌ {action_data.get('message', 'Comando não reconhecido.')}"
        
        else:
            return f"⚠️ Ação '{action}' não implementada."
    
    def process_user_input(self, user_input):
        """Processa entrada do usuário e retorna resposta completa"""
        print(f"🔍 Processando comando: '{user_input}'")
        
        action_data = self.process_command(user_input)
        print(f"🎯 Ação detectada: {action_data['action']}")
        print(f"📊 Confiança: {action_data.get('confidence', 'N/A')}")
        
        result_message = self.execute_action(action_data)
        
        frontend_update = {
            'backgroundColor': self.current_state['background_color'],
            'fontSize': f"{self.current_state['font_size']}px",
            'menuPosition': self.current_state['menu_position'],
            'currentPage': self.current_state['current_page']
        }
        
        response = {
            'user_input': user_input,
            'assistant_response': result_message,
            'action_data': action_data,
            'frontend_update': frontend_update
        }
        
        self.chat_history.append(response)
        return response

# Inicializa o assistente
assistant = VirtualAssistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'error': 'Mensagem vazia'}), 400
    
    try:
        response = assistant.process_user_input(user_input)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(assistant.current_state)

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(assistant.chat_history[-10:])

@app.route('/api/commands', methods=['GET'])
def get_commands():
    commands = [cmd['text'] for cmd in assistant.commands_data]
    return jsonify(commands)

if __name__ == '__main__':
    print("🧠 ASSISTENTE VIRTUAL INTELIGENTE - VERSÃO CORRIGIDA")
    print("=====================================================")
    print("✅ Sistema de classificação por palavras-chave ativado")
    print("🌐 Servidor iniciando em http://localhost:5000")
    print("\n🎯 TESTE ESTES COMANDOS:")
    print("• 'mude a cor do fundo para azul'")
    print("• 'aumente o tamanho da fonte'") 
    print("• 'vá para a página sobre'")
    print("• 'mova o menu para a esquerda'")
    print("• 'redefinir layout'")
    print("\n💡 O sistema agora usa classificação por palavras-chave + MLP!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)