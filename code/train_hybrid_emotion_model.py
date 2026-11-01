import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load data
mat_file = 'de_features_subset.mat'
mat = sio.loadmat(mat_file)
features = mat['all_features'] # shape (3720, 56)
labels_raw = mat['all_labels'] # shape (3720, 4)

# Reshape into trials (120 trials, 31 windows, 56 features)
num_trials = 120
num_windows = 31
num_features = 56

X = features.reshape((num_trials, num_windows, num_features))
# Labels are constant across the trial, so just take the first window's label
y_full = labels_raw.reshape((num_trials, num_windows, 4))[:, 0, :]

# Target: Valence (idx 0), Arousal (idx 1)
# Binarize labels (>= 5 is High(1), < 5 is Low(0))
y_val = (y_full[:, 0] >= 5).astype(np.int64)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_val, dtype=torch.long)

class SpatioTemporalModel(nn.Module):
    def __init__(self):
        super(SpatioTemporalModel, self).__init__()
        # CNN to process spatial-frequency features for each window
        # Input to CNN: (batch * seq_len, 1, channels(14), bands(4))
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 2), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 2), padding=1)
        
        # Output of pool and convs: 32 * 7 * 3 = 672
        self.flatten = nn.Flatten()
        
        # LSTM to process the sequence of features
        self.lstm = nn.LSTM(input_size=672, hidden_size=64, num_layers=1, batch_first=True)
        
        # Fully connected for classification
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Reshape to treat seq_len as batch for CNN
        x = x.view(batch_size * seq_len, 1, 14, 4)
        
        # CNN Forward
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        
        # Reshape back to sequence for LSTM
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM Forward
        out, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        last_out = out[:, -1, :]
        
        # FC Forward
        logits = self.fc(last_out)
        return logits

# Training logic with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

epochs = 30
batch_size = 16
criterion = nn.CrossEntropyLoss()

fold_accuracies = []
all_preds = []
all_targets = []

print("Starting 5-Fold Cross Validation for CNN-LSTM Model (Valence)...")

for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
    print(f"--- Fold {fold+1} ---")
    
    # Standardize data based on training fold
    X_train = X_tensor[train_idx]
    X_test = X_tensor[test_idx]
    y_train = y_tensor[train_idx]
    y_test = y_tensor[test_idx]
    
    mu = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True)
    std[std == 0] = 1e-6
    
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SpatioTemporalModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            outputs = model(b_x)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    fold_preds = []
    fold_targets = []
    with torch.no_grad():
        for b_x, b_y in test_loader:
            outputs = model(b_x)
            _, predicted = torch.max(outputs.data, 1)
            total += b_y.size(0)
            correct += (predicted == b_y).sum().item()
            fold_preds.extend(predicted.numpy())
            fold_targets.extend(b_y.numpy())
            
    acc = 100 * correct / total
    fold_accuracies.append(acc)
    all_preds.extend(fold_preds)
    all_targets.extend(fold_targets)
    print(f"Fold {fold+1} Accuracy: {acc:.2f}%")

print(f"\nAverage Cross-Validation Accuracy: {np.mean(fold_accuracies):.2f}%")

# Generate Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Valence', 'High Valence'], yticklabels=['Low Valence', 'High Valence'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title(f'CNN-LSTM Confusion Matrix (Avg Acc: {np.mean(fold_accuracies):.2f}%)')

os.makedirs('sci_figures', exist_ok=True)
plt.savefig('sci_figures/cnn_lstm_confusion_matrix.png')
print("Saved Confusion Matrix to sci_figures/cnn_lstm_confusion_matrix.png")
