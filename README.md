# Groebner-Basis-Complexity-Prediction
# Исследование корреляции между вычислительной сложностью базиса Грёбнера и характеристиками полиномиального идеала

**Автор:** Наталья Сидорова  
**Год:** 2026

## О проекте

Данный репозиторий содержит код и результаты дипломной работы по исследованию корреляции между вычислительной сложностью нахождения инволютивного базиса Грёбнера и структурными метриками полиномиального идеала.

### Основные компоненты
- Реализация 8 предвычисляемых метрик идеала
- Корреляционный анализ
- Модели машинного обучения (нейронные сети и деревья решений)
- Визуализация результатов

## Структура репозитория

- `data/` — обработанные данные
- `src/` — основные скрипты Python
- `figures/` — графики и тепловые карты
- `results/` — полученные таблицы корреляци
- `models/` — сохранённые модели

## Как запустить

1. Клонируйте репозиторий:
   git clone https://github.com/nasidorova/Groebner-Basis-Complexity-Prediction.git
   cd Groebner-Basis-Complexity-Prediction
2. Установите зависимости:
   pip install -r requirements.txt
3. Запустите скрипты в следующем порядке:
   python src/prepare_dataset_new.py
   python src/Corr.py
   python src/nn_model_better_new.py
   python src/nn_model_time_new.py
   python src/nn_model_memory_new.py
   python src/desicion_tree_new.py
4. Все графики сохранятся в папку figures/
