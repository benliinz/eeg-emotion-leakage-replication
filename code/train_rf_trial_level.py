import scipy.io as sio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import os
import h5py

print("Loading data locally...")
mat_file = 'de_features_all.mat'
start_time = time.time()
with h5py.File(mat_file, 'r') as f:
    features = np.array(f['all_features']).T # shape (39680, 56)
    labels_raw = np.array(f['all_labels']).T # shape (39680, 4)

# Reshape into trials (1280 trials, 31 windows, 56 features)
num_trials = 1280
num_windows = 31
num_features = 56

X = features.reshape((num_trials, num_windows, num_features))
y_full = labels_raw.reshape((num_trials, num_windows, 4))[:, 0, :]
y_val = (y_full[:, 0] >= 5).astype(np.int64)

# For Random Forest, we need a 2D array.
# Let's flatten the 31 windows * 56 features into a single 1736-dim vector per trial
X_flat = X.reshape((num_trials, -1))

print("Starting Random Forest Trial-Level Evaluation (5-Fold CV per Subject)...")
subject_accuracies = []

for sub in range(32):
    start_idx = sub * 40
    end_idx = (sub + 1) * 40
    
    X_sub = X_flat[start_idx:end_idx]
    y_sub = y_val[start_idx:end_idx]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for train_idx, test_idx in kf.split(X_sub):
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y_sub[train_idx], y_sub[test_idx]
        
        # Standardize features
        mu = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1e-6
        X_train = (X_train - mu) / std
        X_test = (X_test - mu) / std
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        acc = accuracy_score(y_test, preds) * 100
        fold_accuracies.append(acc)
        
    sub_acc = np.mean(fold_accuracies)
    subject_accuracies.append(sub_acc)
    print(f"Subject {sub+1} Accuracy: {sub_acc:.2f}%")

global_mean = np.mean(subject_accuracies)
global_std = np.std(subject_accuracies)
print(f"\n--- Final RF Trial-Level Accuracy: {global_mean:.2f}% +/- {global_std:.2f}% ---")

plt.figure(figsize=(12, 6))
bars = plt.bar(range(1, 33), subject_accuracies, color='orange', edgecolor='black')
plt.axhline(y=global_mean, color='red', linestyle='--', linewidth=2, label=f'Mean Accuracy ({global_mean:.2f}%)')
plt.xlabel('Subject ID', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.title('Rigorous Trial-Level Accuracy (Random Forest on DEAP)', fontsize=14)
plt.xticks(range(1, 33))
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.0f}', ha='center', va='bottom', fontsize=8)

os.makedirs('sci_figures', exist_ok=True)
plt.tight_layout()
plt.savefig('sci_figures/rf_trial_level_accuracy.png', dpi=300)
print("Saved RF Accuracy Chart to sci_figures/rf_trial_level_accuracy.png")
