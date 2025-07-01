import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
print(">> Módulo de TrainingArguments:", TrainingArguments.__module__)
import numpy as np
import evaluate

# 1. Carrega o JSON como HuggingFace Dataset
def load_custom_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        conversation = item.get("conversation", [])
        user_utterance = ""
        assistant_response = ""

        for msg in conversation:
            if msg["role"] == "user":
                user_utterance = msg["content"]
            elif msg["role"] == "assistant":
                assistant_response = msg["content"]

        if user_utterance and assistant_response:
            text = f"Usuário: {user_utterance}\nAssistente: {assistant_response}"
            label = 1  # ou outro valor se você quiser uma classificação binária arbitrária
            samples.append({"text": text, "label": label})

    return Dataset.from_list(samples)

# 2. Tokenização
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding=False)

# 3. Métrica (exemplo: accuracy)
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Caminho do seu JSON
json_path = "./data/dialog.json"

# Nome do modelo base
model_name = "distilbert-base-uncased"

# Carrega tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Dataset
dataset = load_custom_dataset(json_path)
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./results_distilbert",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    do_eval=True,  # Garante que o Trainer vai rodar avaliação
    save_steps=500,  # Ou outro valor, obrigatório na versão antiga
    eval_steps=500,  # Mesmo motivo
)

from sklearn.model_selection import train_test_split

# Split manual em train/test
train_data, test_data = dataset.train_test_split(test_size=0.2).values()
tokenized_train = train_data.map(tokenize_fn, batched=True)
tokenized_test = test_data.map(tokenize_fn, batched=True)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

# Inicia o treinamento
trainer.train()
