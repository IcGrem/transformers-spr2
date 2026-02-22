import torch
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional

class TextGenerationEvaluator:
    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Используемое устройство: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # # 3. Исправление для моделей типа GPT-2 (у них нет pad_token по умолчанию)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # # Загрузка метрики ROUGE
        self.rouge_metric = evaluate.load("rouge")

    def generate_texts(self, prompts: List[str], 
                       max_length: int = 100, 
                       num_return_sequences: int = 1,
                       **generation_kwargs) -> List[str]:
        """
        Генерация продолжений текста для списка промптов.
        
        Args:
            prompts: Список входных строк.
            max_length: Максимальная длина сгенерированной последовательности.
            num_return_sequences: Количество вариантов генерации на один промпт.
            **generation_kwargs: Дополнительные аргументы для model.generate (temperature, top_p и т.д.).
            
        Returns:
            Список сгенерированных текстов.
        """
        generated_texts = []
        
        # Токенизация входных данных
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        input_lengths = inputs['input_ids'].shape[1]

        # Генерация
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs
            )

        # Декодирование
        # Важно: output_ids содержат и промпт, и сгенерированную часть.
        # Для ROUGE обычно сравнивают полный текст с полным референсом, 
        # либо только сгенерированную часть с продолжением референса.
        # Здесь мы возвращаем полный сгенерированный текст.
        for i in range(output_ids.shape[0]):
            text = self.tokenizer.decode(output_ids[i], skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Вычисление метрик ROUGE между предсказаниями и референсами.
        
        Args:
            predictions: Список сгенерированных текстов.
            references: Список эталонных текстов (ground truth).
            
        Returns:
            Словарь со значениями метрик (rouge1, rouge2, rougeL).
        """
        if len(predictions) != len(references):
            raise ValueError("Длины списков predictions и references должны совпадать.")

        results = self.rouge_metric.compute(
            predictions=predictions, 
            references=references
        )
        return results # type: ignore

    def validate(self, prompts: List[str], references: List[str], 
                 gen_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Полный цикл валидации: генерация + оценка метрик.
        
        Args:
            prompts: Список промптов для генерации.
            references: Список эталонных текстов для сравнения.
            gen_config: Словарь параметров генерации (max_length, temperature и т.д.).
            
        Returns:
            Словарь с результатами метрик.
        """
        print("Начало генерации текстов...")
        
        # Генерация с прогресс-баром (если нужно, можно разбить на батчи внутри generate_texts)
        # В текущей реализации generate_texts уже обрабатывает список, 
        # но для наглядности добавим tqdm, если список очень большой (опционально)
        predictions = self.generate_texts(prompts, **gen_config)

        print("Вычисление метрик ROUGE...")
        metrics = self.calculate_rouge(predictions, references)

        return metrics

# ==========================================
# Пример использования
# ==========================================

if __name__ == "__main__":
    # 1. Инициализация класса
    # Используем легкую модель для быстрого примера
    evaluator = TextGenerationEvaluator(model_name="distilgpt2")

    # 2. Подготовка данных (Промпты и Референсы)
    # В реальном сценарии references - это полные целевые тексты
    dataset_prompts = [
        "Люблю грозу в начале мая",
        "Когда весна приходит в город",
        "Искусственный интеллект меняет мир"
    ]
    
    # Эталоны (для примера возьмем продолжения, которые мы считаем "правильными")
    # В задаче автодополнения референс обычно включает промпт + идеальное продолжение
    dataset_references = [
        "Люблю грозу в начале мая, Когда весенний первый гром...",
        "Когда весна приходит в город, Люди снимают пальто и шапки...",
        "Искусственный интеллект меняет мир, Делая процессы быстрее и эффективнее..."
    ]

    # 3. Настройка параметров генерации
    generation_params = {
        "max_length": 60,          # Общая длина (промпт + ответ)
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "no_repeat_ngram_size": 2
    }

    # 4. Запуск валидации
    try:
        results = evaluator.validate(
            prompts=dataset_prompts,
            references=dataset_references,
            gen_config=generation_params
        )

        # 5. Вывод результатов
        print('\n--- Результаты метрик ---')
        for k, v in results.items():
            # v может быть float или dict (в зависимости от версии evaluate), обычно float
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        # Пример первого сгенерированного текста
        print('\n--- Пример генерации ---')
        # Сгенерируем отдельно для наглядности
        sample_out = evaluator.generate_texts([dataset_prompts[0]], **generation_params)
        print(f"Промпт: {dataset_prompts[0]}")
        print(f"Генерация: {sample_out[0]}")
        
    except Exception as e:
        print(f"Произошла ошибка при выполнении: {e}")