import torch

from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from src.data_utils import DataUtils, data_utils

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


class NextTokenDataset(Dataset):
    def __init__(self, clean_data, tokenizer, seq_len = 3, tokens_csv: str|None = None):
        # self.samples - список пар (X, Y)
        # X - токенизированный текст
        # Y - токенизированный текст, сдвинутый на один токен
        self.samples = []
        self.tokens = []
        if tokens_csv:
            self._samples_processing_from_csv(tokens_csv, seq_len)
        else:
            self._samples_processing(clean_data, tokenizer, seq_len)


    def __len__(self) -> int:
        return len(self.samples) # верните размер датасета


    def __getitem__(self, idx: int) -> tuple:
        x, y = self.samples[idx] # получите контекст и таргет для элемента с индексом idx
        return torch.tensor(x), torch.tensor(y)


    def _samples_processing(self, clean_data: DataUtils, tokenizer: BertTokenizerFast, seq_len: int = 3):
        print("Start samples processing ->", len(clean_data))
        for text in clean_data:
            token_ids = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True) # токенизированная строка с твитом
            self.tokens.append(token_ids)
            # если строка слишком короткая, то пропускаем её
            if len(token_ids) <= seq_len:
                continue
            self.__context_target(token_ids, seq_len)


    def _samples_processing_from_csv(self, tokens_csv, seq_len: int = 3):
        print("Start samples processing from csv ->", tokens_csv)
        tokens = data_utils.load_tokenized_data_csv(tokens_csv)
        for token_ids in tokens:
            # если строка слишком короткая, то пропускаем её
            if len(token_ids) <= seq_len:
                continue
            self.__context_target(token_ids, seq_len)


    def __context_target(self, token_ids, seq_len):
        # проходимся по всем токенам в последовательности
        for i in range(0, len(token_ids) - seq_len):
            '''
            для context надо из списка взять срез из трёх элементов, т.е. от i до i+3.
            для таргет надо взять срез из трёх элемнтов со смещением +1, т.е. от i+1 до i+4
            '''
            context = token_ids[i: i+3]
            target = token_ids[i+1: i+4]
            self.samples.append((context, target))
