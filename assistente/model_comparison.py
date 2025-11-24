from sklearn.metrics import accuracy_score, f1_score, classification_report

def get_model_metrics(assistant):
    test_data = [
        {"input": "mude a cor do fundo para azul", "expected": "change_background_color"},
        {"input": "vá para a página sobre", "expected": "navigate"},
        {"input": "aumente o tamanho da fonte", "expected": "change_font_size"},
        {"input": "redefinir layout", "expected": "reset_layout"},
        # Adicione mais exemplos para melhorar a avaliação!
    ]
    
    # Use o assistente passado como parâmetro (já treinado)
    results_wv = []
    results_tr = []
    expected = []

    for item in test_data:
        expected.append(item["expected"])
        assistant.switch_model('word2vec')
        out_wv = assistant.process_command(item["input"])
        results_wv.append(out_wv.get("action", "unknown"))
        assistant.switch_model('transformer')
        out_tr = assistant.process_command(item["input"])
        results_tr.append(out_tr.get("action", "unknown"))

    return {
        "word2vec": {
            "accuracy": accuracy_score(expected, results_wv),
            "f1": f1_score(expected, results_wv, average="weighted"),
            "report": classification_report(expected, results_wv, output_dict=True)
        },
        "transformer": {
            "accuracy": accuracy_score(expected, results_tr),
            "f1": f1_score(expected, results_tr, average="weighted"),
            "report": classification_report(expected, results_tr, output_dict=True)
        }
    }