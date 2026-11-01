import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time

print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Load data
mat_file = 'de_features_all.mat'
print(f"Loading data from {mat_file} using h5py...")
start_time = time.time()
with h5py.File(mat_file, 'r') as f:
    features = np.array(f['all_features']).T # shape (39680, 56)
    labels_raw = np.array(f['all_labels']).T # shape (39680, 4)
print(f"Data loaded in {time.time() - start_time:.2f}s")

# Instead of reshaping into (trials, windows, features), 
# we keep it as (samples, features) where each sample is 1 window!
# We just need to separate it per subject.
num_subjects = 32
windows_per_subject = 1240 # 40 trials * 31 windows
num_features = 56

X = features # (39680, 56)
y_val = (labels_raw[:, 0] >= 5).astype(np.int64) # (39680,)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_val, dtype=torch.long)

class StaticCNNModel(nn.Module):
    def __init__(self):
        super(StaticCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 2), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 2), padding=1)
        self.flatten = nn.Flatten()
        
        # Flattened size is 32 * 7 * 3 = 672
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(672, 64)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        # x is (batch, 56)
        x = x.view(-1, 1, 14, 4)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

epochs = 50
batch_size = 64
criterion = nn.CrossEntropyLoss()

subject_accuracies = []

print("Starting Window-Level (Leaked) Evaluation (5-Fold CV per Subject)...")

for sub in range(num_subjects):
    start_idx = sub * windows_per_subject
    end_idx = (sub + 1) * windows_per_subject
    
    X_sub = X_tensor[start_idx:end_idx]
    y_sub = y_tensor[start_idx:end_idx]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_sub)):
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y_sub[train_idx], y_sub[test_idx]
        
        mu = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True)
        std[std == 0] = 1e-6
        X_train = (X_train - mu) / std
        X_test = (X_test - mu) / std
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        
        model = StaticCNNModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                optimizer.zero_grad()
                outputs = model(b_x)
                loss = criterion(outputs, b_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for b_x, b_y in test_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                outputs = model(b_x)
                _, predicted = torch.max(outputs.data, 1)
                total += b_y.size(0)
                correct += (predicted == b_y).sum().item()
        
        fold_accuracies.append(100 * correct / total)
        
    sub_acc = np.mean(fold_accuracies)
    subject_accuracies.append(sub_acc)
    print(f"Subject {sub+1} Accuracy: {sub_acc:.2f}%")

global_mean = np.mean(subject_accuracies)
global_std = np.std(subject_accuracies)
print(f"\n--- Final Window-Level Accuracy (With Leakage): {global_mean:.2f}% +/- {global_std:.2f}% ---")

# Generate Bar Chart of Accuracies
plt.figure(figsize=(12, 6))
bars = plt.bar(range(1, 33), subject_accuracies, color='lightgreen', edgecolor='black')
plt.axhline(y=global_mean, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean Accuracy ({global_mean:.2f}%)')
plt.xlabel('Subject ID', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.title('Window-Level "Data Leakage" Accuracy (CNN on DEAP)', fontsize=14)
plt.xticks(range(1, 33))
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('window_level_accuracy.png', dpi=300)
print("Saved Accuracy Chart to window_level_accuracy.png")
