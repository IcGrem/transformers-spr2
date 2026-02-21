import numpy as np
import re
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast

from src.data_utils import DataUtils, data_utils

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# класс датасета
class NextTokenDataset(Dataset):
    def __init__(self, clean_data, tokenizer, seq_len = 3, tokens_csv: str|None = None):
        # tokens_csv: str|None = None
        # self.samples - список пар (X, Y)
        # X - токенизированный текст
        # Y - токенизированный текст, сдвинутый на один токен
        self.samples = []
        # self.tokens = data_utils.load_tokenized_data_csv(tokens_csv) if tokens_csv else []
        self.tokens = []
        # self.load_tokens = load_tokens
        if tokens_csv:
            self._samples_processing_from_csv(tokens_csv, seq_len)
        else:
            self._samples_processing(clean_data, tokenizer, seq_len)



    def __len__(self) -> int:
        return len(self.samples)# верните размер датасета


    def __getitem__(self, idx: int) -> tuple:
        # print(self.samples)
        x, y = self.samples[idx] # получите контекст и таргет для элемента с индексом idx
        return torch.tensor(x), torch.tensor(y)


    def _samples_processing(self, clean_data: DataUtils, tokenizer: BertTokenizerFast, seq_len: int = 3):
        print("Start samples processing ->", len(clean_data))
        
        for text in clean_data:
            # print(f'text --- {text}')
            token_ids = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True) # токенизированная строка с твитом
            # print(f'token_ids --- {token_ids}')
            self.tokens.append(token_ids)

            # если строка слишком короткая, то пропускаем её
            if len(token_ids) <= seq_len:
                # print(f"len(token_ids) {len(token_ids)} <= seq_len {seq_len}")
                continue
            self.__context_target(token_ids, seq_len)


    def _samples_processing_from_csv(self, tokens_csv, seq_len: int = 3):
        print("Start samples processing from csv ->", tokens_csv)
        tokens = data_utils.load_tokenized_data_csv(tokens_csv)
        for token_ids in tokens:
            # если строка слишком короткая, то пропускаем её
            if len(token_ids) <= seq_len:
                # print(f"len(token_ids) {len(token_ids)} <= seq_len {seq_len}")
                continue
            self.__context_target(token_ids, seq_len)


    def __context_target(self, token_ids, seq_len):
        # проходимся по всем токенам в последовательности
        for i in range(0, len(token_ids) - seq_len):
            '''
            для context надо из списка взять срез из трёх элементов, т.е. от i до i+2.
            для таргет надо взять срез из трёх элемнтов со смещением +1, т.е. от i+1 до i+3
            '''

            context = token_ids[i: i+3] # срез для контекста
            # print(f"i={i}, context-> {context}")
            # если контекст слишком короткий, то пропускаем его
            target = token_ids[i+1: i+4] # возьмите i-ый токен последовательности
            # print(f"i={i}, target-> {target}")
            self.samples.append((context, target))


    # def split_dataset(self):


# # разбиение на тренировочную и валидационную выборки
# val_size = 0.05

# train_texts, val_texts = train_test_split(cleaned_texts, test_size=val_size, random_state=42)
# print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")

# # тренировочный и валидационный датасеты
# train_dataset = MaskedBertDataset(train_texts, tokenizer, seq_len=seq_len)
# val_dataset = MaskedBertDataset(val_texts, tokenizer, seq_len=seq_len)


# # даталоадеры
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64)

# print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
# print(f"Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}")