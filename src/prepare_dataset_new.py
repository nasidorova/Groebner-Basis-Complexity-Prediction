import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os



# 1. Загрузка данных
metrics = pd.read_csv('metrics_only_1.csv', sep=';')
master = pd.read_excel('MasterMetrics.xlsx', sheet_name='ResultsM')

# Нормализация имён
metrics['key'] = metrics['name'].str.replace(r'\.json|,\s*json', '', regex=True).str.strip().str.lower()
master['key']  = master['name'].str.replace(r'\.json|,\s*json', '', regex=True).str.strip().str.lower()

# Merge
dataset = pd.merge(metrics, master[['key', 'time', 'avr memory']], on='key', how='inner')
dataset = dataset.drop(columns=['key'])

print(f"Объединено строк: {len(dataset)}")

# 2. Очистка
numeric_cols = dataset.select_dtypes(include=np.number).columns
dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())

z_scores = np.abs(zscore(dataset[numeric_cols]))
dataset = dataset[(z_scores < 4).all(axis=1)].reset_index(drop=True)

print(f"После очистки: {len(dataset)} строк")

# 3. Разделение на Train / Val / Test (70% / 15% / 15%)
X = dataset.drop(['name', 'time', 'avr memory'], axis=1, errors='ignore')
y = dataset[['time', 'avr memory']]

#  отделяем Test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

#  делим оставшиеся 85% на Train (70%) и Val (15%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42   # 15/85 ≈ 0.1765
)

print(f"Размеры наборов:")
print(f"  Train: {X_train.shape[0]} строк")
print(f"  Val:   {X_val.shape[0]} строк")
print(f"  Test:  {X_test.shape[0]} строк")

# 4. Нормализация  ТОЛЬКО на Train
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# 5. Сохранение в новые файлы
dataset.to_csv('groebner_dataset_clean_new.csv', sep=';', index=False)

np.savez('data_split_new.npz',
         X_train=X_train_scaled,
         X_val=X_val_scaled,
         X_test=X_test_scaled,
         y_train=y_train.values,
         y_val=y_val.values,
         y_test=y_test.values)


print("Созданы файлы:")
print("    groebner_dataset_clean_new.csv")
print("    data_split_new.npz  (содержит X_val и y_val)")