// Estado da aplicação
let currentState = {
    backgroundColor: '#ffffff',
    textColor: '#000000',
    fontSize: '16px',
    menuPosition: 'center',
    currentPage: 'home',
    headerPosition: 'top'
};

// Navegação entre páginas
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', function() {
        const page = this.dataset.page;
        navigateToPage(page);
    });
});


function navigateToPage(page) {
    // Atualiza itens do menu
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.page === page) {
            item.classList.add('active');
        }
    });

    // Atualiza páginas
    document.querySelectorAll('.page-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`page-${page}`).classList.add('active');

    // Atualiza estado
    currentState.currentPage = page;
    updateStateDisplay();
}

// Envia mensagem ao pressionar Enter
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Envia exemplo de comando
function sendExample(command) {
    document.getElementById('userInput').value = command;
    sendMessage();
}

// Envia mensagem para o backend
async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();

    if (!message) return;

    // Adiciona mensagem do usuário no chat
    addMessage('user', message);
    input.value = '';

    // Mostra loading
    document.getElementById('loading').classList.add('active');

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        // Adiciona resposta do assistente
        const confidence = data.action_data.confidence 
            ? `(${(data.action_data.confidence * 100).toFixed(0)}% confiança)` 
            : '';
        
        addMessage('assistant', data.assistant_response + 
            (confidence ? `<span class="confidence-badge">${confidence}</span>` : ''));

        // Atualiza estado visual
        if (data.frontend_update) {
            updateUI(data.frontend_update);
        }

    } catch (error) {
        addMessage('assistant', '❌ Erro ao processar comando: ' + error.message);
    } finally {
        document.getElementById('loading').classList.remove('active');
    }
}

// Adiciona mensagem no chat
function addMessage(sender, text) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const label = sender === 'user' ? 'Você' : 'Assistente';
    messageDiv.innerHTML = `
        <div class="message-label">${label}</div>
        <div>${text}</div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Atualiza interface com base no estado
function updateUI(update) {
    const body = document.getElementById('body');
    const navMenu = document.getElementById('navMenu');
    navMenu.classList.remove('left', 'right', 'center', 'bottom', 'top');
    navMenu.classList.add(update.menuPosition);

    // Adiciona/remover classe para ajuste do body
    if (update.menuPosition === 'bottom') {
        document.body.classList.add('has-bottom-menu');
    } else {
        document.body.classList.remove('has-bottom-menu');
    }
    
    // Atualiza cor de fundo
    if (update.backgroundColor) {
        body.style.backgroundColor = update.backgroundColor;
        currentState.backgroundColor = update.backgroundColor;
    }

    // Atualiza cor do texto
    if (update.textColor) {
        body.style.color = update.textColor;
        currentState.textColor = update.textColor;
    }

    // Atualiza tamanho da fonte
    if (update.fontSize) {
        body.style.fontSize = update.fontSize;
        currentState.fontSize = update.fontSize;
    }

    // Atualiza posição do menu
    if (update.menuPosition) {
        navMenu.className = 'nav-menu ' + update.menuPosition;
        currentState.menuPosition = update.menuPosition;
    }

    // Navega para página
    if (update.currentPage && update.currentPage !== currentState.currentPage) {
        navigateToPage(update.currentPage);
    }

    updateStateDisplay();
}

// Atualiza display do estado
function updateStateDisplay() {
    document.getElementById('bgColorValue').textContent = currentState.backgroundColor;
    document.getElementById('textColorValue').textContent = currentState.textColor;
    document.getElementById('fontSizeValue').textContent = currentState.fontSize;
    document.getElementById('menuPosValue').textContent = currentState.menuPosition;
    document.getElementById('currentPageValue').textContent = currentState.currentPage;
}

// Carrega estado inicial
async function loadInitialState() {
    try {
        const response = await fetch('/api/state');
        const state = await response.json();
        
        const update = {
            backgroundColor: state.background_color,
            textColor: state.text_color,
            fontSize: `${state.font_size}px`,
            menuPosition: state.menu_position,
            currentPage: state.current_page
        };
        
        updateUI(update);
        const modelResponse = await fetch('/api/current_model');
        const modelData = await modelResponse.json();
        document.getElementById('modelSelect').value = modelData.model;
        document.getElementById('currentModelDisplay').innerHTML = `Modelo atual: <strong>${modelData.model === 'word2vec' ? 'Word2Vec' : 'Transformer (Sentence Transformers)'}</strong>`;

    } catch (error) {
        console.error('Erro ao carregar estado:', error);
    }
}

async function changeModel() {
    const model = document.getElementById('modelSelect').value;
    try {
        const response = await fetch('/api/change_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: model })
        });
        const data = await response.json();
        addMessage('assistant', data.message || 'Modelo alterado!');
        
        // Atualiza o display do modelo
        document.getElementById('currentModelDisplay').innerHTML = `Modelo atual: <strong>${model === 'word2vec' ? 'Word2Vec' : 'Transformer (Sentence Transformers)'}</strong>`;
    } catch (error) {
        addMessage('assistant', 'Erro ao mudar modelo: ' + error.message);
    }
}

// Carrega estado ao iniciar
loadInitialState();

document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('inputText');
    const btnCompare = document.getElementById('btnCompare');
    const btnWV = document.getElementById('btnWV');
    const btnTR = document.getElementById('btnTR');
    const wvResult = document.getElementById('wvResult');
    const trResult = document.getElementById('trResult');
    const statusDiv = document.getElementById('statusDiv');

    async function fetchStatus() {
        const res = await fetch('/api/debug/status');
        const data = await res.json();
        statusDiv.textContent = `Current: ${data.current_model} — Training: ${JSON.stringify(data.training_status)}`;
        return data;
    }

    async function compareText() {
        const text = inputText.value.trim();
        if (!text) return;
        const url = `/api/debug/compare?text=${encodeURIComponent(text)}`;
        const res = await fetch(url);
        const data = await res.json();
        if (data.error) {
            alert(data.error);
            return;
        }
        const result = data.result;
        const labels = data.labels || {};

        const render = (modelName, out) => {
            const mlp = out.mlp;
            const mlpPred = out.mlp_prediction;
            const top = out.top_similar || [];
            return {
                model: modelName,
                vector_dim: out.vector_dim,
                mlp_ready: mlp.ready,
                mlp_pred_index: mlpPred.class,
                mlp_pred_action: mlpPred.action || (mlpPred.class !== null ? labels[mlpPred.class] : null),
                mlp_confidence: mlpPred.confidence,
                top_similar: top
            };
        };

        const wv = render('word2vec', result.word2vec);
        const tr = render('transformer', result.transformer);

        wvResult.textContent = JSON.stringify(wv, null, 2);
        trResult.textContent = JSON.stringify(tr, null, 2);

        await fetchStatus();
    }

    async function changeModel(model) {
        await fetch('/api/change_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model })
        });
        await fetchStatus();
    }

    btnCompare.addEventListener('click', compareText);
    btnWV.addEventListener('click', () => changeModel('word2vec'));
    btnTR.addEventListener('click', () => changeModel('transformer'));

    fetchStatus();
});