import json
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from tqdm import tqdm
import pandas as pd

class BioBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize input text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def parse_evidences2(evidences, json_data):
    """
    Parse evidences into meaningful text using JSON mappings.
    Combine repeated questions' answers with '&' and separate antecedents.
    """
    parsed_antecedents = {}
    parsed_symptoms = {}

    for evidence in eval(evidences):  # Convert string list to actual list
        if "_@_" in evidence:
            code, value = evidence.split("_@_")
            question = json_data.get(code, {}).get('question_en', 'Unknown question')
            value_meaning = json_data.get(code, {}).get('value_meaning', {}).get(value, {}).get('en', value)
            is_antecedent = json_data.get(code, {}).get('is_antecedent', False)
            target_dict = parsed_antecedents if is_antecedent else parsed_symptoms
            if question in target_dict:
                target_dict[question] += f" & {value_meaning}"
            else:
                target_dict[question] = value_meaning
        else:
            question = json_data.get(evidence, {}).get('question_en', 'Unknown question')
            is_antecedent = json_data.get(evidence, {}).get('is_antecedent', False)
            target_dict = parsed_antecedents if is_antecedent else parsed_symptoms
            if question in target_dict:
                target_dict[question] += " & Y"
            else:
                target_dict[question] = "Y"

    antecedents = [f"{q} - {a}" for q, a in parsed_antecedents.items()]
    symptoms = [f"{q} - {a}" for q, a in parsed_symptoms.items()]
    return antecedents, symptoms

def transform_data(csv_path, json_path):
    """
    Transforms the CSV and JSON data into BioBERT-friendly format.
    """
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    csv_data = pd.read_csv(csv_path)
    formatted_data = []

    for _, row in csv_data.iterrows():
        antecedents, symptoms = parse_evidences2(row['EVIDENCES'], json_data)
        patient_data = {
            "input_text": " ".join(antecedents + symptoms),
            "label": row['DIFFERENTIAL_DIAGNOSIS'].split(',')[0].strip()  # Use the primary diagnosis
        }
        formatted_data.append(patient_data)

    return formatted_data

def prepare_datasets(train_data, val_data, test_data, all_labels, tokenizer, max_length):
    """
    Prepares datasets and label encoder.
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    def encode_dataset(data):
        texts = [entry["input_text"] for entry in data]
        labels = label_encoder.transform([entry["label"] for entry in data])
        return BioBERTDataset(texts, labels, tokenizer, max_length)

    train_dataset = encode_dataset(train_data)
    val_dataset = encode_dataset(val_data)
    test_texts = [entry["input_text"] for entry in test_data]  # Test set doesn't have labels

    return train_dataset, val_dataset, test_texts, label_encoder

def train_model(train_dataset, val_dataset, model_path="dmis-lab/biobert-v1.1", max_length=128, batch_size=16, epochs=3, lr=2e-5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(set(train_dataset.labels)))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch["label"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_predictions, val_true_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                attention_mask = batch["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                labels = batch["label"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                val_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        val_f1 = f1_score(val_true_labels, val_predictions, average="weighted")

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained("best_biobert_model")
            tokenizer.save_pretrained("best_biobert_model")
            print("Best model saved!")

    return model

def evaluate_model(test_texts, label_encoder, model_path="best_biobert_model", max_length=128, batch_size=16):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    test_dataset = BioBERTDataset(test_texts, [0] * len(test_texts), tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            logits = model(input_ids, attention_mask=attention_mask).logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

    predicted_labels = label_encoder.inverse_transform(predictions)
    return predicted_labels

# File paths
train_csv = "dataset/samples/train_sample50.csv"
val_csv = "dataset/samples/val_sample20.csv"
test_csv = "dataset/samples/test_sample50.csv"
json_path = "release_evidences_cleaned.json"

# Full list of available labels
all_labels = json.load(open('true_labels.json'))

# Transform data
train_data = transform_data(train_csv, json_path)
val_data = transform_data(val_csv, json_path)
test_data = transform_data(test_csv, json_path)

print(val_data)

# # Prepare datasets
# tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
# train_dataset, val_dataset, test_texts, label_encoder = prepare_datasets(train_data, val_data, test_data, all_labels, tokenizer, max_length=128)

# # Train model
# model = train_model(train_dataset, val_dataset)

# # Evaluate model
# predicted_labels = evaluate_model(test_texts, label_encoder)
# print(predicted_labels)
