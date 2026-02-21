import numpy as np
import re
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# класс датасета
class TweetsDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=3):
        # self.samples - список пар (X, Y)
        # X - токенизированный текст
        # Y - токенизированный текст, сдвинутый на одно слово
        self.samples = []
        for text in texts:
            print(f'text --- {text}')
            token_ids = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True) # токенизированная строка с твитом
            print(f'token_ids --- {token_ids}')
            # если строка слишком короткая, то пропускаем её
            if len(token_ids) < seq_len:
                print(f"len(token_ids) {len(token_ids)} < seq_len {seq_len}")
                continue
            # проходимся по всем токенам в последовательности
            for i in range(0, len(token_ids) - 3):
                '''
                context - список из seq_len // 2 токенов до i-го токена, токена tokenizer.mask_token_id, и seq_len // 2 токенов после i-го токена

                для context надо из списка взять срез из трёх элементов, т.е. от i до i+2.
                для таргет надо взять срез из трёх элемнтов со смещением +1, т.е. от i+1 до i+3
                '''

                context = token_ids[i: i+3] # срез для контекста
                print(f"i={i}, context-> {context}")
                # если контекст слишком короткий, то пропускаем его
                # if len(context) < seq_len:
                    # continue
                target = token_ids[i+1: i+4] # возьмите i-ый токен последовательности
                print(f"i={i}, target-> {target}")
                self.samples.append((context, target))

           
    def __len__(self):
        return len(self.samples)# верните размер датасета


    def __getitem__(self, idx):
        print(self.samples)
        x, y = self.samples[idx] # получите контекст и таргет для элемента с индексом idx
        return torch.tensor(x), torch.tensor(y)


class LoadClean:
    def __init__(self):
        self.data = []
    
    def clean_text(self, text):
        text = text.lower()                 # Приведение к нижнему регистру
        text = re.sub(r'[^a-zа-яё0-9\s]', '', text) # Удаление спецсимволов (regex)
        text = re.sub(r"\s+", " ", text).strip()       # Удаление лишних пробелов
        return text

    def load_and_clean(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            cleaned_text = self.clean_text(line)
            if cleaned_text:
                self.data.append(cleaned_text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
    

