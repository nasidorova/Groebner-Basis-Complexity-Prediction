import pandas as pd

print(" ДИАГНОСТИКА ИМЁН ")

# 1. Загружаем файлы
metrics = pd.read_csv('metrics_only_1.csv', sep=';')
master  = pd.read_excel('MasterMetrics.xlsx', sheet_name='КорреляцииВСЁ')

print(f"metrics_only_1.csv: {len(metrics)} строк")
print("Первые 5 имён в metrics: ", metrics['name'].head(5).tolist())
print(f"MasterMetrics.xlsx: {len(master)} строк")
print("Первые 5 имён в Master: ", master['name'].head(5).tolist())

#  ИСПРАВЛЕНИЕ КЛЮЧЕЙ
# Убираем и ".json", и ",json", и возможные пробелы
metrics['key'] = (metrics['name']
                  .str.replace(r'\.json|,\s*json', '', regex=True)
                  .str.strip()
                  .str.lower())

master['key'] = (master['name']
                 .str.replace(r'\.json|,\s*json', '', regex=True)
                 .str.strip()
                 .str.lower())

#  ОБЪЕДИНЕНИЕ
df_merged = pd.merge(
    metrics,
    master[['key', 'time', 'avr memory']],
    on='key',
    how='inner'
)

print(f"\nУспешно объединено строк: {len(df_merged)}")

if len(df_merged) == 0:
    print("Всё равно не совпало!")
    exit()

# Убираем временный ключ
df = df_merged.drop(columns=['key'])

#  КОРРЕЛЯЦИИ
cols = [
    'time', 'avr memory',
    'dimension', 'density', 'total_monomials', 'max_degree_input',
    'd_reg_asymp', 'first_fall_estimate',
    'non_mult_prolongations_initial', 'macaulay_bound'
]

numeric_df = df[cols].select_dtypes(include='number')
corr_matrix = numeric_df.corr(method='pearson')


print("Топ корреляций с time:")
print(corr_matrix['time'].sort_values(ascending=False).round(4))

print("\nТоп корреляций с avr memory:")
print(corr_matrix['avr memory'].sort_values(ascending=False).round(4))

print("\nПолная матрица (округлено):")
print(corr_matrix.round(4))

#
corr_matrix.round(4).to_csv('corr_table.csv', sep=';', index=True)
print('\nТаблица корреляций сохранена в corr_table.csv')

df.to_excel('MasterMetrics_updated.xlsx', index=False)
print('Объединённый датасет сохранён в MasterMetrics_updated.xlsx')