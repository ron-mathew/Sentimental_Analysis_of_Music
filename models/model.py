import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, Dataset
import random

# Define Dataset Class for Lyrics
class LyricsDataset(Dataset):
    def __init__(self, lyrics_texts, labels, tokenizer, max_length=128):
        self.lyrics_texts = lyrics_texts
        self.labels = labels  # Emotion labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lyrics_texts)

    def __getitem__(self, idx):
        text = self.lyrics_texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        inputs = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': inputs,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define the RHYME Model Architecture
class RHYMEModel(nn.Module):
    def __init__(self, num_emotions=5):
        super(RHYMEModel, self).__init__()
        config = BertConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        
        self.bert = BertModel(config)
        self.emotion_layer = nn.Linear(config.hidden_size, num_emotions)  # Emotion classification head
        self.mlm_layer = nn.Linear(config.hidden_size, config.vocab_size)  # MLM head for masked tokens

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Emotion classification
        pooled_output = outputs.pooler_output  # [CLS] token's output
        emotion_logits = self.emotion_layer(pooled_output)
        
        # MLM Prediction
        sequence_output = outputs.last_hidden_state  # Sequence of hidden states
        mlm_logits = self.mlm_layer(sequence_output)

        return emotion_logits, mlm_logits

# Initialize Model, Tokenizer, Optimizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = RHYMEModel(num_emotions=5)  # Assume 5 emotion classes for this example
optimizer = AdamW(model.parameters(), lr=2e-5)

# Example Lyrics Data (Replace this with your dataset)
lyrics_texts = ["I feel so alive and free", "Lonely nights keep haunting me"]
labels = [1, 0]  # Binary emotions for demonstration (1=positive, 0=negative)

# Prepare DataLoader
dataset = LyricsDataset(lyrics_texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define Training Loop
epochs = 3
loss_fn = nn.CrossEntropyLoss()  # Emotion classification loss

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        emotion_labels = batch['label']
        
        # Forward pass
        emotion_logits, mlm_logits = model(input_ids, attention_mask)
        
        # Emotion classification loss
        emotion_loss = loss_fn(emotion_logits, emotion_labels)
        
        # Total loss (could add MLM loss here if masked tokens are available)
        total_loss = emotion_loss
        total_loss.backward()  # Backpropagation
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch + 1} completed with loss {total_loss.item()}")

# Evaluation Code (Optional)
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            emotion_labels = batch['label']
            
            # Forward pass
            emotion_logits, _ = model(input_ids, attention_mask)
            
            # Calculate accuracy
            _, predicted = torch.max(emotion_logits, dim=1)
            correct += (predicted == emotion_labels).sum().item()
            total += emotion_labels.size(0)
    
    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")

# Run Evaluation
evaluate(model, dataloader)
