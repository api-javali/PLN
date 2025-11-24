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
    // Verifica se navMenu existe antes de acessar classList
    if (navMenu) {
        navMenu.classList.remove('left', 'right', 'center', 'bottom', 'top');
        navMenu.classList.add(update.menuPosition);
        navMenu.className = 'nav-menu ' + update.menuPosition;
    }

    // Verifica se body existe antes de acessar classList
    if (body) {
        if (update.menuPosition === 'bottom') {
            body.classList.add('has-bottom-menu');
        } else {
            body.classList.remove('has-bottom-menu');
        }
        if (update.backgroundColor) body.style.backgroundColor = update.backgroundColor;
        if (update.textColor) body.style.color = update.textColor;
        if (update.fontSize) body.style.fontSize = update.fontSize;
    }

    // Navega para página
    if (update.currentPage && update.currentPage !== currentState.currentPage) {
        navigateToPage(update.currentPage);
    }

    // Atualiza estado
    currentState.backgroundColor = update.backgroundColor || currentState.backgroundColor;
    currentState.textColor = update.textColor || currentState.textColor;
    currentState.fontSize = update.fontSize || currentState.fontSize;
    currentState.menuPosition = update.menuPosition || currentState.menuPosition;
    currentState.currentPage = update.currentPage || currentState.currentPage;

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
    const compareInput = document.getElementById('compareInput');
    const btnCompareModels = document.getElementById('btnCompareModels');
    const compareWvResult = document.getElementById('compareWvResult');
    const compareTrResult = document.getElementById('compareTrResult');
    const compareSummary = document.getElementById('compareSummary');

    async function compareText() {
        const text = compareInput.value.trim();
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

        // Renderiza resultado Word2Vec
        const wv = result.word2vec;
        const wvPred = wv.mlp_prediction;
        compareWvResult.textContent =
            `Ação: ${wvPred.action || labels[wvPred.class]}\n` +
            `Confiança: ${(wvPred.confidence * 100).toFixed(1)}%\n` +
            `Similaridade: ${wv.top_similar.length ? (wv.top_similar[0].similarity * 100).toFixed(1) + '%' : '—'}\n` +
            `Dimensão do vetor: ${wv.vector_dim}`;

        // Renderiza resultado Transformer
        const tr = result.transformer;
        const trPred = tr.mlp_prediction;
        compareTrResult.textContent =
            `Ação: ${trPred.action || labels[trPred.class]}\n` +
            `Confiança: ${(trPred.confidence * 100).toFixed(1)}%\n` +
            `Similaridade: ${tr.top_similar.length ? (tr.top_similar[0].similarity * 100).toFixed(1) + '%' : '—'}\n` +
            `Dimensão do vetor: ${tr.vector_dim}`;

        // Resumo visual
        if ((wvPred.action || labels[wvPred.class]) === (trPred.action || labels[trPred.class])) {
            compareSummary.textContent = "AÇÕES IGUAIS entre modelos";
            compareSummary.style.color = "green";
        } else {
            compareSummary.textContent = "AÇÕES DIFERENTES entre modelos";
            compareSummary.style.color = "red";
        }
    }

    btnCompareModels.addEventListener('click', compareText);

   const btnLoadMetrics = document.getElementById('btnLoadMetrics');
    const metricsResult = document.getElementById('metricsResult');

    async function loadMetrics() {
        metricsResult.textContent = "Carregando...";
        const res = await fetch('/api/model_metrics');
        const data = await res.json();
        metricsResult.textContent =
            `Word2Vec:\nAcurácia: ${(data.word2vec.accuracy * 100).toFixed(1)}%\nF1: ${(data.word2vec.f1 * 100).toFixed(1)}%\n\n` +
            `Transformer:\nAcurácia: ${(data.transformer.accuracy * 100).toFixed(1)}%\nF1: ${(data.transformer.f1 * 100).toFixed(1)}%`;
    }

    btnLoadMetrics.addEventListener('click', loadMetrics);
});