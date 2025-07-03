import torch
import pickle
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json # Importa a biblioteca json

def load_inference_components(model_path, binarizer_path):
    """
    Carrega o tokenizador, o modelo e o MultiLabelBinarizer salvos.
    """
    print(f"Carregando tokenizador de: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Carregando modelo de: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Coloca o modelo em modo de avaliação (importante para inferência)
    model.eval()

    print(f"Carregando MultiLabelBinarizer de: {binarizer_path}")
    with open(binarizer_path, "rb") as f:
        mlb = pickle.load(f)
    return tokenizer, model, mlb

def predict_labels(text, tokenizer, model, mlb, threshold=0.5):
    """
    Realiza a inferência em um texto e retorna os rótulos preditos.

    Args:
        text (str): O texto de entrada para classificação.
        tokenizer: O tokenizador do modelo.
        model: O modelo treinado.
        mlb: O MultiLabelBinarizer para decodificar os rótulos.
        threshold (float): O limiar de probabilidade para considerar um rótulo como presente.

    Returns:
        list: Uma lista de rótulos preditos para o texto.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    predicted_indices = np.where(probs > threshold)[0]
    binary_prediction_vector = np.zeros(len(mlb.classes_))
    binary_prediction_vector[predicted_indices] = 1
    predicted_labels = mlb.inverse_transform(binary_prediction_vector.reshape(1, -1))

    return list(predicted_labels[0])

# --- FUNÇÃO PARA CARREGAR OS DADOS DO SEU ARQUIVO DE INFERÊNCIA ---
def load_inference_data(json_path):
    """
    Carrega os dados do arquivo JSON de inferência e extrai as mensagens de usuário
    com task_id == 4.
    """
    inference_samples = []
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for item in raw_data:
        conversation = item.get("conversation", [])
        user_msg = ""
        # Percorre a conversa para encontrar a mensagem do usuário com task_id 4
        for msg in conversation:
            if msg.get("role") == "user" and msg.get("task_id") == 4:
                user_msg = msg["content"]
                break # Encontrou, pode sair do loop interno

        if user_msg:
            # Você pode adicionar um ID se o seu arquivo original tiver um,
            # ou gerar um sequencialmente. Aqui, vamos usar o índice.
            inference_samples.append({"original_item": item, "text": user_msg})
        else:
            print(f"Aviso: Não foi encontrada mensagem de usuário com task_id 4 no item: {item}. Pulando.")
    return inference_samples

# --- Configurações de Inferência ---
MODEL_DIR = "./results_task4_distilbert" # Caminho onde o modelo foi salvo
BINARIZER_PATH = "multi_label_binarizer.pkl" # Caminho onde o binarizer foi salvo
INFERENCE_FILE_PATH = "./data/dialog4inference_train_part.json" # <<<<<< Configure o caminho do seu arquivo de inferência aqui

# Carrega os componentes para inferência
tokenizer, model, mlb = load_inference_components(MODEL_DIR, BINARIZER_PATH)

# --- Realizando Inferência no arquivo personalizado ---
print(f"\n--- Realizando Inferência no arquivo: {INFERENCE_FILE_PATH} ---")
inference_data_to_process = load_inference_data(INFERENCE_FILE_PATH)

results = []
# Use o mesmo limiar que você experimentou no treinamento (por exemplo, 0.3)
inference_threshold = 0.3 # <<<<<< Ajuste o limiar para a inferência aqui

if not inference_data_to_process:
    print("Nenhum dado válido foi carregado do arquivo de inferência. Verifique o formato do arquivo.")
else:
    for i, item in enumerate(inference_data_to_process):
        text_content = item["text"]
        original_item_context = item["original_item"] # Mantém o contexto original para referência

        predicted_labels = predict_labels(text_content, tokenizer, model, mlb, threshold=inference_threshold)
        
        # Adiciona os resultados a uma lista, mantendo o contexto original se útil
        results.append({
            "index": i,
            "original_content": text_content, # O texto que foi classificado
            "predicted_labels": predicted_labels,
            "original_json_item": original_item_context # Se quiser guardar o item completo original
        })

# Opcional: Salvar os resultados da inferência em um novo arquivo JSON
output_inference_path = "./inference_results.json"
with open(output_inference_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"\nResultados da inferência salvos em: {output_inference_path}")

# Opcional: Imprimir alguns resultados para visualização imediata
print("\n--- Primeiros Resultados da Inferência ---")
for res in results[:5]: # Imprime os 5 primeiros resultados
    print(f"Texto: '{res['original_content']}'")
    print(f"Rótulos Preditos: {res['predicted_labels']}\n")