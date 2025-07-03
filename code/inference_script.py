import torch
import pickle
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

def predict_labels(text, tokenizer, model, mlb, threshold=0.3):
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
    # Tokeniza o texto de entrada
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move os inputs para a GPU se disponível
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model.to('cuda')

    # Faz a predição sem calcular gradientes
    with torch.no_grad():
        outputs = model(**inputs)

    # Extrai os logits (saídas brutas do modelo)
    logits = outputs.logits

    # Aplica a função sigmoide para obter probabilidades (0 a 1) para cada rótulo
    probs = torch.sigmoid(logits).cpu().numpy()[0] # .cpu() para mover para CPU antes de converter para numpy

    # Converte probabilidades em predições binárias usando o limiar
    # np.where(condition, if_true, if_false)
    # Retorna os índices onde a probabilidade é maior que o limiar
    predicted_indices = np.where(probs > threshold)[0]

    # Cria um vetor binário com 1s nas posições dos rótulos preditos
    # e 0s nas demais, com o mesmo tamanho do número total de classes.
    binary_prediction_vector = np.zeros(len(mlb.classes_))
    binary_prediction_vector[predicted_indices] = 1

    # Usa o MultiLabelBinarizer para transformar o vetor binário de volta para os nomes dos rótulos
    # Note que inverse_transform espera uma lista de vetores, por isso reshape(1, -1)
    predicted_labels = mlb.inverse_transform(binary_prediction_vector.reshape(1, -1))

    # inverse_transform retorna uma lista de tuplas, pegamos o primeiro elemento da primeira tupla
    return list(predicted_labels[0])

# --- Configurações de Inferência ---
MODEL_DIR = "./results_task4_distilbert" # Caminho onde o modelo foi salvo
BINARIZER_PATH = "multi_label_binarizer.pkl" # Caminho onde o binarizer foi salvo

# Carrega os componentes para inferência
tokenizer, model, mlb = load_inference_components(MODEL_DIR, BINARIZER_PATH)

# --- Exemplos de Teste ---
test_texts = [
    "Brazil needs help after the big flood in the south. Donate!",
    "Report from CNN: A huge hurricane is approaching Florida. Stay safe.",
    "Small earthquake felt in Tokyo, no significant damage reported.",
    "People in rural areas are requesting food and water after the severe drought.",
    "There was a minor tremor in the Andes. All is well.",
    "The volcano erupted, spewing ash for miles. Evacuation orders are in place."
]

print("\n--- Realizando Inferência nos Textos de Exemplo ---")
for i, text in enumerate(test_texts):
    predicted = predict_labels(text, tokenizer, model, mlb)
    print(f"\nTexto {i+1}: '{text}'")
    print(f"Rótulos Preditos: {predicted}")

# --- Testando com seu próprio arquivo de inferência ---
# Se você tem um arquivo JSON com textos para inferir (ex: {"text": "...", "id": ...})
# Adapte esta seção para ler seu arquivo.

# Exemplo de arquivo de inferência JSON (infer_data.json):
# [
#     {"id": 1, "text": "RT emartillo1: Ecuador is in an urgent need of international help..donate and share the information of the donnation account.Thanks :"},
#     {"id": 2, "text": "A severe blizzard hit the mountains, causing power outages."}
# ]

# import json
#
# INFERENCE_FILE_PATH = "./data/infer_data.json" # Supondo que seu arquivo esteja aqui
#
# print(f"\n--- Realizando Inferência no arquivo: {INFERENCE_FILE_PATH} ---")
# with open(INFERENCE_FILE_PATH, "r", encoding="utf-8") as f:
#     inference_data = json.load(f)
#
# results = []
# for item in inference_data:
#     text_id = item.get("id")
#     text_content = item.get("text")
#     if text_content:
#         predicted_labels = predict_labels(text_content, tokenizer, model, mlb)
#         results.append({"id": text_id, "text": text_content, "predicted_labels": predicted_labels})
#     else:
#         print(f"Aviso: Item sem 'text' encontrado (ID: {text_id}). Pulando.")
#
# # Opcional: Salvar os resultados da inferência em um novo arquivo JSON
# output_inference_path = "./inference_results.json"
# with open(output_inference_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=4, ensure_ascii=False)
# print(f"\nResultados da inferência salvos em: {output_inference_path}")