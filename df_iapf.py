from typing import List, Tuple
import os
import re
import json
from collections import defaultdict, Counter

from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

from conf import EXTRACTED_PATH, TITLE_PATH


def sentence_split(text: str) -> List[str]:
    text = re.sub(r'\[\*\*.+?\*\*\]', '', text.strip())
    text = re.sub(r'\n\n+|  +', '\t', text)
    sp = re.sub(r'\n\n+|  +', '\t', text.strip())\
           .replace('\n', ' ')\
           .replace('!', '\t')\
           .replace('?', '\t')\
           .replace('.', '\t')\
           .replace(':', '\t')
    sentences = [s.strip() for s in sp.split('\t') if s.strip()]
    return sentences


class DFIAPF:
    def __init__(self, dataset: List[str], n: int):
        tokenizer = RegexpTokenizer(r'\w+')
        self.dataset = []
        for document in tqdm(dataset):
            sentences = []
            for sentence in sentence_split(document):
                words = tokenizer.tokenize(sentence.lower())
                if len(words) <= n:
                    sentences.append(words)
            self.dataset.append(sentences)
        self.n = n
        self.df_iapf = {}

    def get_n_grams(self, sentence: List[str], n: int) -> List[str]:
        n_grams = []
        for i in range(len(sentence) - n + 1):
            n_grams.append(' '.join(sentence[i:(i + n)]))
        return n_grams

    def calculate_n(self, dataset_n_grams: List[str], n: int) -> None:
        df_pf = defaultdict(lambda: [0, 0])
        for n_grams in tqdm(dataset_n_grams, desc=f'calculating df_iapf for {n} gram'):
            cur_pf = Counter(n_grams)
            for phrase, count in cur_pf.items():
                record = df_pf[phrase]
                record[0] += 1
                record[1] += count
        for phrase, record in df_pf.items():
            self.df_iapf[phrase] = record[0] ** 2 / (record[1] * len(dataset_n_grams))

    def calculate(self, pre_rank_k, rank_k, return_all_titles=True) -> List[Tuple[str, float]]:
        for i in range(1, self.n + 1):
            dataset_n_grams = []
            for document in tqdm(self.dataset, desc=f'counting {i} gram'):
                document_n_grams = []
                for sentence in document:
                    sentence_n_grams = self.get_n_grams(sentence, i)
                    document_n_grams.extend(sentence_n_grams)
                dataset_n_grams.append(document_n_grams)
            self.calculate_n(dataset_n_grams, i)
        df_iapf = sorted(self.df_iapf.items(), key=lambda e: e[1], reverse=True)

        pre_rank_titles = df_iapf[:pre_rank_k]
        filtered_titles = []
        for i, (r, s) in enumerate(pre_rank_titles):
            for j, (rr, ss) in enumerate(pre_rank_titles):
                if i != j and r in rr:
                    break
            else:
                filtered_titles.append(r)
        titles = filtered_titles if rank_k == -1 else filtered_titles[:rank_k]
        if return_all_titles:
            return titles, df_iapf
        return titles


if __name__ == '__main__':
    if not os.path.exists(TITLE_PATH) or not os.path.isdir(TITLE_PATH):
        os.makedirs(TITLE_PATH)
    train_data = json.load(open(os.path.join(EXTRACTED_PATH, 'mimic3_train.json'), 'r', encoding='utf-8'))
    dataset = [patient['text'] for patient in train_data]
    df_iapf_calculator = DFIAPF(dataset, n=5)
    titles, all_titles = df_iapf_calculator.calculate(pre_rank_k=100, rank_k=23, return_all_titles=True)
    open(os.path.join(TITLE_PATH, 'titles.txt'), 'w', encoding='utf-8').writelines([f'{title}\n' for title in titles])
    open(os.path.join(TITLE_PATH, 'all_titles.txt'), 'w', encoding='utf-8').writelines([f'{title}, {score}\n' for title, score in all_titles])
