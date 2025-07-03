# train_task4_distilbert.py

import json
import pickle
from datasets import Dataset
import ast # Para avaliar strings com segurança como expressões Python
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from functools import partial

print(">> Módulo da TrainingArguments em uso:", TrainingArguments.__module__)
import transformers
print("Transformers version:", transformers.__version__)

# 1. Filtra task_id == 4 e prepara o dataset
def load_task4_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        conversation = item.get("conversation", [])
        user_msg = ""
        assistant_response = ""

        # Encontra a mensagem do usuário e a resposta do assistente associadas ao task_id 4
        for i in range(len(conversation)):
            if conversation[i].get("role") == "user" and conversation[i].get("task_id") == 4:
                user_msg = conversation[i]["content"]
            if i > 0 and conversation[i-1].get("task_id") == 4 and conversation[i].get("role") == "assistant":
                assistant_response = conversation[i]["content"]

        if user_msg and assistant_response:
            try:
                # Avalia a string com segurança para um dicionário Python
                labels_dict = ast.literal_eval(assistant_response.strip())
                # Extrai todos os valores do dicionário, tratando listas e valores únicos
                extracted_labels = []
                for key, value in labels_dict.items():
                    if isinstance(value, list):
                        # Filtra para garantir que apenas strings não-vazias e não-None sejam adicionadas
                        extracted_labels.extend([v for v in value if isinstance(v, str) and v.strip() != ''])
                    else:
                        # Filtra para garantir que apenas strings não-vazias e não-None sejam adicionadas
                        if isinstance(value, str) and value.strip() != '':
                            extracted_labels.append(value)
                        # Opcional: Adicione um print para depurar se houver valores inesperados
                        # elif value is None:
                        #     print(f"Aviso: Encontrado valor None para o rótulo '{key}'. Ignorando.")
                        # else:
                        #     print(f"Aviso: Encontrado valor inesperado '{value}' para o rótulo '{key}'. Ignorando.")

                # Adiciona a amostra apenas se houver rótulos válidos extraídos
                if extracted_labels:
                    samples.append({"text": user_msg, "labels": extracted_labels})
                else:
                    print(f"Aviso: Nenhuns rótulos válidos extraídos para a entrada: '{user_msg}' com resposta '{assistant_response}'. Ignorando esta amostra.")

            except (ValueError, SyntaxError) as e:
                print(f"Pulando resposta do assistente malformada: {assistant_response} - Erro: {e}")
                continue # Pula amostras com rótulos malformados

    return Dataset.from_list(samples)

# O restante do seu código permanece o mesmo

# 2. Carrega o dataset
json_path = "./data/dialog.json"
dataset = load_task4_dataset(json_path)

# 3. Codifica os rótulos para Multi-Label
all_labels = [label for sample in dataset for label in sample["labels"]]
# Verifique se all_labels está vazio aqui. Se estiver, significa que nenhuma amostra
# com rótulos válidos foi carregada.
if not all_labels:
    raise ValueError("Não foram encontrados rótulos válidos no dataset. Verifique o seu arquivo JSON.")

mlb = MultiLabelBinarizer()
mlb.fit([all_labels]) # Ajusta ao conjunto de todos os rótulos possíveis

# Transforma os rótulos no dataset
def binarize_labels(example, mlb_fitted):
    return {"labels": mlb_fitted.transform([example["labels"]])[0].astype(float)}

dataset = dataset.map(partial(binarize_labels, mlb_fitted=mlb))

# Salva o MultiLabelBinarizer para inferência futura
with open("multi_label_binarizer.pkl", "wb") as f:
    pickle.dump(mlb, f)

# 4. Tokeniza
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("labels", "label")
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# 5. Divide o dataset
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 6. Modelo
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)

# 7. Métricas de avaliação para multi-label
def compute_metrics(p):
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    probs = torch.sigmoid(torch.tensor(predictions)).numpy()
    preds = np.where(probs > 0.5, 1, 0)

    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    roc_auc_micro = roc_auc_score(labels, probs, average="micro")
    roc_auc_macro = roc_auc_score(labels, probs, average="macro")
    exact_match_ratio = accuracy_score(labels, preds)

    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "roc_auc_micro": roc_auc_micro,
        "roc_auc_macro": roc_auc_macro,
        "exact_match_ratio": exact_match_ratio,
    }
    return metrics

# 8. Treinamento
training_args = TrainingArguments(
    output_dir="./results_task4_distilbert",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./results_task4_distilbert")

print("Avaliando o modelo no conjunto de teste:")
eval_results = trainer.evaluate(eval_dataset)
print(eval_results)