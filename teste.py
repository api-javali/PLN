import requests, json, time

BASE = "http://localhost:5000"
SAMPLES = [
    "mude a cor do fundo para azul",
    "coloque o menu à esquerda",
    "aumente o tamanho da fonte",
    "vá para a página de contatos",
    "redefinir layout",
    "mude a cor do texto para vermelho",
    "fundo laranja",
    "diminua a fonte",
    "menu ao centro",
    "quero ver sobre"
]

def compare(text):
    r = requests.get(f"{BASE}/api/debug/compare", params={'text': text})
    return r.json()

def status():
    return requests.get(f"{BASE}/api/debug/status").json()

if __name__ == '__main__':
    print("Status inicial:", json.dumps(status(), indent=2))
    for model in ['word2vec','transformer']:
        # pede change_model e espera classifier
        print(f"\nSwitching to {model} ...")
        requests.post(f"{BASE}/api/change_model", json={'model': model})
        # aguarda pronto até 10s
        for _ in range(20):
            s = status()
            if s['training_status'].get(model) == 'ready':
                break
            time.sleep(0.5)
        print("Status:", status()['training_status'][model])

        for text in SAMPLES:
            out = compare(text)
            print(f"\n{text}\n -> vector_dim: {out[model]['vector_dim']}, mlp_ready: {out[model]['mlp']['ready']}, mlp_pred: {out[model]['mlp_prediction']}, top_similar: {out[model]['top_similar'][:1]}")