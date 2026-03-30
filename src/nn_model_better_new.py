import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# ПУТЬ ДЛЯ СОХРАНЕНИЯ ГРАФИКОВ
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Графики будут сохранены в папку: {FIGURES_DIR}")

#  ЗАГРУЗКА ДАННЫХ
NEW_NPZ = 'data_split_new.npz'
if not os.path.exists(NEW_NPZ):
    print(f"Error: {NEW_NPZ} not found. Run prepare_dataset_new.py first.")
    exit()

print(f"Загружаем новый датасет: {NEW_NPZ}")
data = np.load(NEW_NPZ)

X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

# Log transform
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)

# Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_log, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val_log, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_log, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


#  МОДЕЛЬ
class GroebnerPredictor(nn.Module):
    def __init__(self, input_size=8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


model = GroebnerPredictor()
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

#  ОБУЧЕНИЕ с Validation
epochs = 300
patience = 50
best_loss = float('inf')
counter = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Early stopping на Validation set
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model_new.pt')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stop at epoch {epoch + 1}')
            break

#  ФИНАЛЬНАЯ ОЦЕНКА НА TEST
model.load_state_dict(torch.load('best_model_new.pt'))
model.eval()

with torch.no_grad():
    preds_log = model(X_test_t).numpy()

preds = np.expm1(preds_log)
y_test_orig = np.expm1(y_test_log)

print("\nMETRICS ON ORIGINAL SCALE (new split) ")
print(
    f"Time   - Test MSE: {mean_squared_error(y_test_orig[:, 0], preds[:, 0]):.4f} | R2: {r2_score(y_test_orig[:, 0], preds[:, 0]):.4f}")
print(
    f"Memory - Test MSE: {mean_squared_error(y_test_orig[:, 1], preds[:, 1]):.4f} | R2: {r2_score(y_test_orig[:, 1], preds[:, 1]):.4f}")


#  CONFUSION MATRIX + HEATMAP
def assign_classes(y_values):
    df_y = pd.DataFrame(y_values, columns=['value'])
    low = df_y['value'].quantile(0.33)
    med = df_y['value'].quantile(0.66)
    return np.where(df_y['value'] <= low, 0,
                    np.where(df_y['value'] <= med, 1, 2))


# Time
y_test_time_cl = assign_classes(y_test_orig[:, 0])
pred_time_cl = assign_classes(preds[:, 0])
cm_time = confusion_matrix(y_test_time_cl, pred_time_cl)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_time, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix — Better Model (Time, new)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix_time_new.png'), dpi=300)
plt.show()

print(f'Time Test F1-score (weighted): {f1_score(y_test_time_cl, pred_time_cl, average="weighted"):.4f}')

# Memory
y_test_mem_cl = assign_classes(y_test_orig[:, 1])
pred_mem_cl = assign_classes(preds[:, 1])
cm_mem = confusion_matrix(y_test_mem_cl, pred_mem_cl)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_mem, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix — Better Model (Memory, new)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix_memory_new.png'), dpi=300)
plt.show()

print(f'Memory Test F1-score (weighted): {f1_score(y_test_mem_cl, pred_mem_cl, average="weighted"):.4f}')

#  LOSS CURVE
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss — Better Model (new)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'loss_curve_better_new.png'), dpi=300)
plt.show()

#  СОХРАНЕНИЕ МОДЕЛИ
torch.save(model.state_dict(), 'groebner_predictor_better_new.pt')
print(f'\nНовая модель сохранена как groebner_predictor_better_new.pt')
print(f'Все новые графики сохранены в папке: {FIGURES_DIR}')