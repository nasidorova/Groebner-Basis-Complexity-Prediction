import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# ПУТЬ ДЛЯ СОХРАНЕНИЯ ГРАФИКОВ
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Графики будут сохранены в папку: {FIGURES_DIR}")

#  ЗАГРУЗКА НОВОГО ДАТАСЕТА
NEW_DATASET = 'groebner_dataset_clean_new.csv'
if not os.path.exists(NEW_DATASET):
    print(f"Error: {NEW_DATASET} not found. Run prepare_dataset_new.py first.")
    exit()

print(f"Загружаем новый датасет: {NEW_DATASET}")
df = pd.read_csv(NEW_DATASET, sep=';')

# Убираем regularity_approx
if 'regularity_approx' in df.columns:
    df = df.drop(columns=['regularity_approx'])
    print("Метрика 'regularity_approx' удалена")

print(f"Размер датасета: {df.shape}")

#  СОЗДАНИЕ КЛАССОВ
time_low = df['time'].quantile(0.33)
time_med = df['time'].quantile(0.66)
df['class_time'] = np.where(df['time'] <= time_low, 0,
                            np.where(df['time'] <= time_med, 1, 2))

mem_low = df['avr memory'].quantile(0.33)
mem_med = df['avr memory'].quantile(0.66)
df['class_memory'] = np.where(df['avr memory'] <= mem_low, 0,
                              np.where(df['avr memory'] <= mem_med, 1, 2))

# ПОДГОТОВКА
X = df.drop(['name', 'time', 'avr memory', 'class_time', 'class_memory'], axis=1, errors='ignore')
y_time = df['class_time']
y_mem = df['class_memory']

X_train, X_test, y_time_train, y_time_test = train_test_split(
    X, y_time, test_size=0.2, random_state=42, stratify=y_time
)

_, _, y_mem_train, y_mem_test = train_test_split(
    X, y_mem, test_size=0.2, random_state=42, stratify=y_mem
)

#  ФУНКЦИЯ ВИЗУАЛИЗАЦИИ
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.show()

#  DECISION TREE
print("\nDECISION TREE (new dataset) ")

dt_time = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_time.fit(X_train, y_time_train)
dt_time_pred = dt_time.predict(X_test)

print("DT Time Accuracy:", accuracy_score(y_time_test, dt_time_pred))
print(classification_report(y_time_test, dt_time_pred))

cm_dt_time = confusion_matrix(y_time_test, dt_time_pred)
plot_confusion_matrix(cm_dt_time, 'Confusion Matrix — Decision Tree (Time, new)', 'confusion_dt_time_new.png')
print("DT Time F1-score (weighted):", f1_score(y_time_test, dt_time_pred, average='weighted'))

# Feature Importance для Time
importances = pd.Series(dt_time.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.plot(kind='bar')
plt.title('Feature Importance — Decision Tree (Time, new)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance_time_new.png'), dpi=300)
plt.show()

pickle.dump(dt_time, open('dt_time_new.pkl', 'wb'))

# Memory
dt_mem = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_mem.fit(X_train, y_mem_train)
dt_mem_pred = dt_mem.predict(X_test)

print("\nDT Memory Accuracy:", accuracy_score(y_mem_test, dt_mem_pred))
print(classification_report(y_mem_test, dt_mem_pred))

cm_dt_mem = confusion_matrix(y_mem_test, dt_mem_pred)
plot_confusion_matrix(cm_dt_mem, 'Confusion Matrix — Decision Tree (Memory, new)', 'confusion_dt_memory_new.png')
print("DT Memory F1-score (weighted):", f1_score(y_mem_test, dt_mem_pred, average='weighted'))

# Feature Importance для Memory
importances_mem = pd.Series(dt_mem.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances_mem.plot(kind='bar')
plt.title('Feature Importance — Decision Tree (Memory, new)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance_memory_new.png'), dpi=300)
plt.show()

pickle.dump(dt_mem, open('dt_memory_new.pkl', 'wb'))

#  KNN
print("\nKNN (new dataset) ")

knn_time = KNeighborsClassifier(n_neighbors=5)
knn_time.fit(X_train, y_time_train)
knn_time_pred = knn_time.predict(X_test)

print("KNN Time Accuracy:", accuracy_score(y_time_test, knn_time_pred))
print(classification_report(y_time_test, knn_time_pred))

cm_knn_time = confusion_matrix(y_time_test, knn_time_pred)
plot_confusion_matrix(cm_knn_time, 'Confusion Matrix — KNN (Time, new)', 'confusion_knn_time_new.png')
print("KNN Time F1-score (weighted):", f1_score(y_time_test, knn_time_pred, average='weighted'))

pickle.dump(knn_time, open('knn_time_new.pkl', 'wb'))

knn_mem = KNeighborsClassifier(n_neighbors=5)
knn_mem.fit(X_train, y_mem_train)
knn_mem_pred = knn_mem.predict(X_test)

print("\nKNN Memory Accuracy:", accuracy_score(y_mem_test, knn_mem_pred))
print(classification_report(y_mem_test, knn_mem_pred))

cm_knn_mem = confusion_matrix(y_mem_test, knn_mem_pred)
plot_confusion_matrix(cm_knn_mem, 'Confusion Matrix — KNN (Memory, new)', 'confusion_knn_memory_new.png')
print("KNN Memory F1-score (weighted):", f1_score(y_mem_test, knn_mem_pred, average='weighted'))

pickle.dump(knn_mem, open('knn_memory_new.pkl', 'wb'))

print(f"\nВсе новые модели и графики сохранены в папке: {FIGURES_DIR}")