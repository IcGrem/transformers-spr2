# transformers-spr2

Проект по генерации следующего токена в тексте (next-token prediction) и сравнению двух подходов:
- собственная LSTM-модель на PyTorch;
- готовая `transformers`-модель для генерации и расчета ROUGE.

Основная работа ведется в ноутбуке `solution.ipynb`, а логика вынесена в модули из `src/` и `configs/`.

## Стек

- Python 3.10+
- PyTorch (`torch`)
- Hugging Face: `transformers`, `datasets`, `evaluate`
- Классический ML: `scikit-learn`, `numpy`, `pandas`, `scipy`
- NLP-утилиты: `nltk`, `rouge_score`
- Визуализация: `matplotlib`
- Конфигурация: `pydantic-settings`
- Среда экспериментов: Jupyter (`ipykernel`, `ipython`)

## Структура проекта

- `solution.ipynb` — основной ноутбук с запуском пайплайна обучения/оценки.
- `configs/config.py` — конфиг проекта (гиперпараметры, параметры генерации, лимиты оценки).
- `src/` — код моделей, датасета, обучения, валидации и оценки через `transformers`.
- `text_processor.py` — ранняя/черновая версия классов подготовки данных.
- `models/` — веса модели и чекпоинты обучения (`.pth`).
- `loss_and_accuracy.png`, `ppl_and_accuracy.png` — графики метрик после обучения.

## Классы и их ответственность


### `src/data_utils.py`

- `DataUtils`  
  Утилитный класс для работы с текстовыми данными:
  - очистка текста (`clean_text`);
  - загрузка и очистка сырых строк из файла (`load_and_clean`);
  - сохранение очищенных данных (`save_cleaned_data`);
  - разделение на train/val/test (`split_data`);
  - сохранение/загрузка токенизированных последовательностей в CSV (`save_tokenized_data_csv`, `load_tokenized_data_csv`);
  - базовая статистика длины текстов (`texts_stats`).

### `src/next_token_dataset.py`

- `NextTokenDataset`  
  `torch.utils.data.Dataset` для задачи next-token prediction.
  Из текста (или заранее сохраненных токенов из CSV) формирует пары:
  - `context` — окно длиной `seq_len` токенов;
  - `target` — то же окно, сдвинутое на 1 токен вперед.  
  Возвращает `torch.tensor(context), torch.tensor(target)` для обучения модели предсказывать следующий токен на каждом шаге окна.

### `src/lstm_model.py`

- `LSTMRNNClassifier`  
  Нейросеть для предсказания следующего токена:
  - `Embedding` для токенов;
  - `LSTM` (однонаправленная);
  - `Dropout` + `LayerNorm`;
  - линейная проекция в размер словаря (`Linear(hidden_dim -> vocab_size)`).
  
  Также содержит методы генерации:
  - greedy (`_generate_ids`);
  - sampling с `temperature` и `top_k` (`_generate_with_sampling`);
  - пользовательский метод `generate(...)`, который декодирует токены в текст.

### `src/lstm_train.py`

- `LSTMTrain`  
  Класс цикла обучения:
  - инициализирует оптимизатор (`Adam`) и функцию потерь (`CrossEntropyLoss`);
  - выполняет эпохи обучения, backprop и gradient clipping;
  - вызывает валидатор (`evaluator.evaluate(...)`);
  - считает и сохраняет метрики (`train_loss`, `val_loss`, `val_accuracy`, `val_ppl`);
  - сохраняет чекпоинт после каждой эпохи в `models/checkpoint-*.pth`.

### `src/eval_lstm.py`

- `EvalLSTM`  
  Класс валидации LSTM:
  - считает `loss` и token-level `accuracy`;
  - опционально считает ROUGE через `evaluate`, генерируя продолжение на основе части входной последовательности;
  - возвращает `(avg_loss, accuracy, avg_rouge)`.

### `src/eval_transformer_pipeline.py`

- `EvalTransformer`  
  Обертка для оценки готовой causal LM из `transformers`:
  - загружает токенизатор и модель по `model_name`;
  - генерирует тексты (`generate_texts`);
  - считает ROUGE (`calculate_rouge`);
  - запускает полный цикл валидации (`validate`).

### `configs/config.py`

- `Settings`  
  Класс конфигурации на базе `BaseSettings`: хранит пути, гиперпараметры LSTM (`EMBEDDING_DIM`, `HIDDEN_DIM`, `NUM_LAYERS`, `DROPOUT`), параметры обучения (`BATCH_SIZE`, `EPOCH`), а также параметры генерации/оценки (`TEMPERATURE`, `TOP_K`, `MAX_ROUGE_SAMPLES`, `MAX_GEN_LENGTH`).  
  Читает переменные из `.env`, если файл присутствует.

## Быстрый старт

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Подготовить данные и запустить обучение/оценку через `solution.ipynb`.

