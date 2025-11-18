from flask import Flask, render_template, request, jsonify
from assistente.virtual_assistant import VirtualAssistant

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
