from flask import Flask, render_template, request, jsonify
from assistente.virtual_assistant import VirtualAssistant
from assistente.model_comparison import get_model_metrics

app = Flask(__name__)

# Inicializa o assistente (treina os modelos na inicialização)
print("\n🚀 Inicializando Assistente Virtual...")
assistant = VirtualAssistant()
print("✅ Assistente pronto!\n")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint para processar comandos do usuário"""
    data = request.get_json()
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'error': 'Mensagem vazia'}), 400
    
    try:
        response = assistant.process_user_input(user_input)
        return jsonify(response)
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/state', methods=['GET'])
def get_state():
    """Retorna o estado atual da interface"""
    return jsonify(assistant.current_state)


@app.route('/api/history', methods=['GET'])
def get_history():
    """Retorna histórico de conversas (últimas 10)"""
    return jsonify(assistant.chat_history[-10:])


@app.route('/api/commands', methods=['GET'])
def get_commands():
    """Lista todos os comandos disponíveis"""
    commands = [cmd['text'] for cmd in assistant.commands_data]
    return jsonify(commands)


@app.route('/api/vocabulary', methods=['GET'])
def get_vocabulary():
    """Retorna o vocabulário do Word2Vec"""
    if assistant.word_embedding.model:
        vocab = list(assistant.word_embedding.model.wv.index_to_key)
        return jsonify({
            'vocabulary': vocab[:100],  # Primeiras 100 palavras
            'total_words': len(vocab)
        })
    return jsonify({'error': 'Modelo não treinado'}), 500


@app.route('/api/similar/<word>', methods=['GET'])
def get_similar_words(word):
    """Encontra palavras similares a uma palavra dada"""
    similar = assistant.word_embedding.most_similar(word, topn=5)
    if similar:
        return jsonify({
            'word': word,
            'similar': [{'word': w, 'score': float(s)} for w, s in similar]
        })
    return jsonify({'error': f'Palavra "{word}" não encontrada'}), 404

@app.route('/api/current_model', methods=['GET'])
def get_current_model():
    return jsonify({'model': assistant.embedding_model_type})

@app.route('/api/change_model', methods=['POST'])
def change_model():
    data = request.get_json()
    new_model = data.get('model', 'word2vec')
    
    if new_model not in ['word2vec', 'transformer']:
        return jsonify({'error': 'Modelo inválido'}), 400
    
    message = assistant.switch_model(new_model)
    return jsonify({'message': message})

@app.route('/api/debug/labels', methods=['GET'])
def debug_labels():
    """Retorna mapping de índices para nomes de ação"""
    return jsonify(assistant.label_decoder)

@app.route('/api/debug/compare', methods=['GET'])
def debug_compare():
    """
    Compara os modelos Word2Vec e Transformer para um texto informado.
    Retorna predição do MLP, similaridade e parâmetros extraídos.
    """
    text = request.args.get('text', '')
    if not text:
        return jsonify({'error': 'Texto vazio'}), 400

    result = {}
    labels = assistant.label_decoder

    for model in ['word2vec', 'transformer']:
        assistant.switch_model(model, background=False)
        embedding = assistant.word_embedding
        mlp = assistant.mlp

        # Vetor do texto
        vector = embedding.text_to_vector(text)
        vector_dim = len(vector) if hasattr(vector, '__len__') else embedding.vector_size

        # MLP
        mlp_ready = mlp is not None
        if mlp_ready:
            pred_class, confidence = mlp.predict_class(vector)
            pred_action = labels.get(pred_class)
        else:
            pred_class, confidence, pred_action = None, 0.0, None

        # Similaridade
        similar_cmd, similarity = assistant.find_most_similar_command(text)
        top_similar = []
        if similar_cmd:
            top_similar.append({
                'text': similar_cmd['text'],
                'action': similar_cmd['action'],
                'similarity': similarity
            })

        result[model] = {
            'vector_dim': vector_dim,
            'mlp': {'ready': mlp_ready},
            'mlp_prediction': {
                'class': pred_class,
                'confidence': confidence,
                'action': pred_action
            },
            'top_similar': top_similar
        }

    return jsonify({'result': result, 'labels': labels})

from assistente.model_comparison import get_model_metrics

@app.route('/api/model_metrics')
def model_metrics():
    try:
        metrics = get_model_metrics(assistant)  # Passe o assistente global
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 ASSISTENTE VIRTUAL - WORD2VEC + MLP")
    print("="*60)
    print("✅ Word2Vec treinado com Gensim")
    print("✅ MLP com backpropagation")
    print("✅ Sistema híbrido inteligente")
    print("✅ Sugestões por similaridade cosseno")
    print("\n🌐 Servidor Flask iniciando...")
    print("📍 URL: http://localhost:5000")
    print("\n🎯 COMANDOS DE TESTE:")
    print("   • 'Olá, mude a cor do fundo para azul'")
    print("   • 'aumente o tamanho da fonte'")
    print("   • 'vá para a página sobre'")
    print("   • 'mova o menu para a esquerda'")
    print("   • 'redefinir layout'")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
