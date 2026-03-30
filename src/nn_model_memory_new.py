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
from sklearn.linear_model import Ridge

#  ПУТЬ ДЛЯ СОХРАНЕНИЯ ГРАФИКОВ
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Графики будут сохранены в папку: {FIGURES_DIR}")

#  ЗАГРУЗКА НОВЫХ ДАННЫХ
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

# Только память (столбец 1)
y_train_mem = np.log1p(y_train[:, 1])
y_val_mem = np.log1p(y_val[:, 1])
y_test_mem = np.log1p(y_test[:, 1])

# Топ-фичи для памяти (после удаления regularity_approx)
top_indices = [0, 3, 4, 5, 7]  # dimension, max_degree_input, d_reg_asymp, first_fall_estimate, macaulay_bound
X_train = X_train[:, top_indices]
X_val = X_val[:, top_indices]
X_test = X_test[:, top_indices]

print(f"Используем {X_train.shape[1]} топ-фич для памяти")

# Аугментация (только на train)
noise = np.random.normal(0, 0.005, X_train.shape)
X_train_aug = np.vstack([X_train, X_train + noise])
y_train_aug = np.concatenate([y_train_mem, y_train_mem])

# Tensors
X_train_t = torch.tensor(X_train_aug, dtype=torch.float32)
y_train_t = torch.tensor(y_train_aug, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val_mem, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_mem, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


#  МОДЕЛЬ
class GroebnerPredictorMemory(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


model = GroebnerPredictorMemory()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

#  ОБУЧЕНИЕ с Validation
epochs = 400
patience = 60
best_loss = float('inf')
counter = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step()

    # Validation для Early Stopping
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t).squeeze()
        val_loss = criterion(val_outputs, y_val_t).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.5f}, Val Loss: {val_loss:.5f}')

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model_memory_new.pt')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stop at epoch {epoch + 1}')
            break

# ФИНАЛЬНАЯ ОЦЕНКА НА TEST
model.load_state_dict(torch.load('best_model_memory_new.pt'))
model.eval()

with torch.no_grad():
    preds_test = model(X_test_t).squeeze().numpy()

preds_test = np.expm1(preds_test)
y_test_orig = np.expm1(y_test_mem)

print("\n MEMORY MODEL RESULTS (new split) ")
print(f"Test MSE:  {mean_squared_error(y_test_orig, preds_test):.4f}")
print(f"Test R2:   {r2_score(y_test_orig, preds_test):.4f}")


# CONFUSION MATRIX + HEATMAP
def assign_classes(y_values):
    df_y = pd.DataFrame(y_values, columns=['value'])
    low = df_y['value'].quantile(0.33)
    med = df_y['value'].quantile(0.66)
    return np.where(df_y['value'] <= low, 0,
                    np.where(df_y['value'] <= med, 1, 2))


y_test_cl = assign_classes(y_test_orig)
pred_cl = assign_classes(preds_test)

cm = confusion_matrix(y_test_cl, pred_cl)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix — Memory Model (new)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix_memory_1_new.png'), dpi=300)
plt.show()

print(f'Memory Test F1-score (weighted): {f1_score(y_test_cl, pred_cl, average="weighted"):.4f}')

#  RIDGE BASELINE
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_aug, y_train_aug)
ridge_preds = ridge.predict(X_test)
ridge_preds = np.expm1(ridge_preds)
print(f'Ridge Test R2: {r2_score(y_test_orig, ridge_preds):.4f}')

#  LOSS CURVE
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss — Memory Model (new)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'loss_curve_memory_new.png'), dpi=300)
plt.show()

#  СОХРАНЕНИЕ МОДЕЛИ
torch.save(model.state_dict(), 'groebner_predictor_memory_new.pt')
print(f'\nНовая модель сохранена как groebner_predictor_memory_new.pt')
print(f'Все новые графики сохранены в папке: {FIGURES_DIR}')