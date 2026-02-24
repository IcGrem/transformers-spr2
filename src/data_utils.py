import csv
import numpy as np
import re

from sklearn.model_selection import train_test_split


class DataUtils:
    def __init__(self):
        self.data = []
    

    def clean_text(self, text: str):
        text = text.lower() #  приведение к нижнему регистру
        text = re.sub(r'[^a-zа-яё0-9\s]', '', text) #  удаление спецсимволов
        text = re.sub(r"\s+", " ", text).strip() #  удаление лишних пробелов
        return text


    def load_and_clean(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            cleaned_text = self.clean_text(line)
            if cleaned_text:
                self.data.append(cleaned_text)
    

    def save_cleaned_data(self, data: list, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data))


    def split_data(self, data: list):
        # делим данные на обучающую 80% и временную 20% выборки
        X_train, X_temp = train_test_split(data, test_size=0.2, random_state=42)
        # делим временную выборку на валидационную 10% и тестовую 10%
        X_valid, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

        print(f"Train data: {len(X_train)}, Val data: {len(X_valid)}, Test data: {len(X_test)}")
        return X_train, X_valid, X_test


    def save_tokenized_data_csv(self, tokenized_data: list, file_path: str):
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(tokenized_data)


    def load_tokenized_data_csv(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            data = [[int(item) for item in row] for row in reader]
        
        return data


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int|slice):
        text = self.data[idx]
        return text


    def texts_stats(self):
        texts = [text for text in self.data]
        word_counts = [len(text.split()) for text in texts]
        print("\nСтатистика по количеству слов в тексте:")
        print(f"Среднее: {np.mean(word_counts):.2f}")
        print(f"Медиана: {np.median(word_counts):.2f}")
        print(f"5-й перцентиль: {np.percentile(word_counts, 5):.2f}")
        print(f"95-й перцентиль: {np.percentile(word_counts, 95):.2f}")


data_utils = DataUtils()