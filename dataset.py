import os
import json
import random

from torch.utils.data import Dataset

from conf import SECTION_PATH
from similarity import build_label_trees, calc_label_similarity


class ContrastiveDataset(Dataset):
    def __init__(self):
        self.selected_titles = [
            'discharge diagnosis',
            'major surgical or invasive procedure',
            'history of present illness',
            'past medical history',
            'brief hospital course',
            'chief complaint',
            'physical exam',
            'discharge medications',
            'discharge disposition',
            'medications on admission',
            'discharge instructions',
            'followup instructions'
        ]

        self.dataset = self.load_dataset()
        self.label_trees = build_label_trees()
        self.indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        s_i = self.sample_s_i(index)
        title_1, title_2 = s_i['title_1'], s_i['title_2']
        s_j = self.sample_s_j(index, title_1, title_2)
        texts = {
            'text_1_i': s_i['text_1_i'],
            'text_2_i': s_i['text_2_i'],
            'text_1_j': s_j['text_1_j'],
            'text_2_j': s_j['text_2_j']
        }
        label_tree_1, all_nodes_1 = self.label_trees[self.dataset[index]['hadm_id']]
        label_tree_2, all_nodes_2 = self.label_trees[self.dataset[s_j['index']]['hadm_id']]
        similarity = calc_label_similarity(label_tree_1, all_nodes_1, label_tree_2, all_nodes_2)
        loss_weight = {
            'w_1i_2i': 1 if len(texts['text_1_i']) > 0 and len(texts['text_2_i']) > 0 else 0,
            'w_1i_1j': 1 if len(texts['text_1_i']) > 0 and len(texts['text_1_j']) > 0 else 0,
            'w_1j_2j': 1 if len(texts['text_1_j']) > 0 and len(texts['text_2_j']) > 0 else 0,
            'w_2i_2j': 1 if len(texts['text_2_i']) > 0 and len(texts['text_2_j']) > 0 else 0
        }
        return texts, similarity, loss_weight

    def load_dataset(self):
        raw_dataset = json.load(open(os.path.join(SECTION_PATH, 'mimic3_train.json'), 'r', encoding='utf-8'))
        dataset = []
        for sample in raw_dataset:
            for title in self.selected_titles:
                text = sample['sections'][title]
                if not self.is_empty_text(text):
                    break
            else:
                continue
            dataset.append(sample)
        return dataset

    def sample_s_i(self, index):
        sections = self.dataset[index]['sections']
        non_empty_sections = []
        for title in self.selected_titles:
            text = sections[title]
            if not self.is_empty_text(text):
                non_empty_sections.append((title, text))
        if len(non_empty_sections) == 1:
            selected_sections = (non_empty_sections[0], (None, ''))
        else:
            selected_sections = random.sample(non_empty_sections, k=2)
        return {
            'title_1': selected_sections[0][0],
            'text_1_i': selected_sections[0][1],
            'title_2': selected_sections[1][0],
            'text_2_i': selected_sections[1][1]
        }
        
    def sample_s_j(self, anchor_index, title_1, title_2):
        neighbor_indices = random.sample(self.indices, k=2)
        neighbor_index = neighbor_indices[0] if neighbor_indices[0] != anchor_index else neighbor_indices[1]
        neighbor_sections = self.dataset[neighbor_index]['sections']
        return {
            'index': neighbor_index,
            'text_1_j': neighbor_sections[title_1] if not self.is_empty_text(neighbor_sections[title_1]) else '',
            'text_2_j': neighbor_sections[title_2] if (title_2 is not None and not self.is_empty_text(neighbor_sections[title_2])) else ''
        }

    def is_empty_text(self, text):
        result = len(text) == 0 or text == 'none'
        return result


class MaskedSectionDataset(Dataset):
    def __init__(self, task, version, mask_rate=0.2):
        self.dataset = self.load_dataset(task, version)
        self.task = task
        self.version = version
        self.mask_rate = mask_rate

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.get_text(index)

    def get_text(self, index):
        if self.version == 'train':
            sections = self.dataset[index]['sections']
            kept_sections = []
            for content in sections.values():
                if len(content) > 0:
                    rate = random.random()
                    if rate >= self.mask_rate:
                        kept_sections.append(content)
            if len(kept_sections) == 0:
                text = ' '.join(sections.values())
            else:
                random.shuffle(kept_sections)
                text = ' '.join(kept_sections)
        else:
            text = self.dataset[index]['text']
        return text

    def load_dataset(self, task, version):
        dataset = json.load(open(os.path.join(SECTION_PATH, f'{task}_{version}.json'), 'r', encoding='utf-8'))
        return dataset


if __name__ == '__main__':
    from tqdm import tqdm

    ds = ContrastiveDataset()
    for i in tqdm(range(len(ds))):
        _ = ds[i]

    ds = MaskedSectionDataset(task='mimic3', version='train', mask_rate=0.2)
    for i in tqdm(range(len(ds))):
        _ = ds[i]
