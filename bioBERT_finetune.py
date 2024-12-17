import json
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from tqdm import tqdm
import numpy as np

class BioBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, multi_label_binarizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlb = multi_label_binarizer

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
            "label": torch.tensor(label, dtype=torch.float)  # Change to float for multi-label
        }

def transform_data_from_json(json_path):
    """
    Transforms the JSON data into BioBERT-friendly format.
    """
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    formatted_data = []

    for entry in json_data:
        antecedents = entry.get('Antecedents', [])
        symptoms = entry.get('Symptoms', [])
        diagnoses = entry.get('Differential Diagnosis', [])

        input_text = " ".join(antecedents + symptoms)
        label = [diag.strip() for diag in diagnoses]

        formatted_data.append({
            "input_text": input_text,
            "label": label
        })

    return formatted_data

def prepare_datasets(train_data, val_data, test_data, all_labels, tokenizer, max_length):
    """
    Prepares datasets and multi-label binarizer.
    """
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels])

    def encode_dataset(data):
        texts = [entry["input_text"] for entry in data]
        labels = [entry["label"] for entry in data]
        
        # Transform labels to binary matrix
        encoded_labels = mlb.transform(labels)

        return texts, encoded_labels

    # Process datasets
    train_texts, train_labels = encode_dataset(train_data)
    val_texts, val_labels = encode_dataset(val_data)
    test_texts = [entry["input_text"] for entry in test_data]

    # Create datasets
    train_dataset = BioBERTDataset(train_texts, train_labels, tokenizer, max_length, mlb)
    val_dataset = BioBERTDataset(val_texts, val_labels, tokenizer, max_length, mlb)

    return train_dataset, val_dataset, test_texts, mlb

def train_model(train_dataset, val_dataset, model_path="dmis-lab/biobert-v1.1", max_length=128, batch_size=16, epochs=3, lr=2e-5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Determine number of labels from the dataset
    num_labels = train_dataset.labels.shape[1]
    
    # Load pre-trained model with correct number of labels
    model = BertForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=num_labels, 
        problem_type="multi_label_classification"
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        "linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_predictions, val_true_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()

                # For multi-label, use sigmoid and threshold
                logits = torch.sigmoid(outputs.logits)
                val_pred = (logits > 0.5).float().cpu().numpy()
                val_predictions.extend(val_pred)
                val_true_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # Convert to numpy for metrics calculation
        val_predictions = np.array(val_predictions)
        val_true_labels = np.array(val_true_labels)

        # Compute metrics
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        val_f1 = f1_score(val_true_labels, val_predictions, average='micro')

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained("best_biobert_model")
            tokenizer.save_pretrained("best_biobert_model")
            print("Best model saved!")

    return model

def evaluate_model(test_texts, multi_label_binarizer, model_path="best_biobert_model", max_length=128, batch_size=16):
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Prepare test dataset
    test_dataset = BioBERTDataset(
        test_texts, 
        np.zeros((len(test_texts), len(multi_label_binarizer.classes_))), 
        tokenizer, 
        max_length, 
        multi_label_binarizer
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get logits and apply sigmoid
            logits = model(input_ids, attention_mask=attention_mask).logits
            predictions = torch.sigmoid(logits)
            
            # Convert to binary predictions
            binary_predictions = (predictions > 0.5).float().cpu().numpy()
            all_predictions.extend(binary_predictions)

    # Convert predictions back to labels
    predicted_labels = multi_label_binarizer.inverse_transform(all_predictions)
    return predicted_labels

# File paths
train_data = "dataset_processed/json/train_sample50.json"
val_data = "dataset_processed/json/val_sample20.json"
test_data = "dataset_processed/json/test_sample50.json"

# Full list of available labels
all_labels = json.load(open('true_labels.json'))

# Transform data
train_data = transform_data_from_json(train_data)
val_data = transform_data_from_json(val_data)
test_data = transform_data_from_json(test_data)

# Prepare datasets
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
train_dataset, val_dataset, test_texts, label_encoder = prepare_datasets(
    train_data, val_data, test_data, all_labels, tokenizer, max_length=128
)

# Train model
model = train_model(train_dataset, val_dataset)

# Uncomment to evaluate
# predicted_labels = evaluate_model(test_texts, label_encoder)
# print(predicted_labels)