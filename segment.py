import os
import json

from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
import gensim

from conf import MIMIC3_PATH, EXTRACTED_PATH, TITLE_PATH, SECTION_PATH


def find_index(words, name_words, names):
    len_name_words = len(name_words)
    for i in range(len(words) - len_name_words):
        candidate = words[i:(i + len_name_words)]
        if candidate == name_words:
            if i == 0:
                return i
            if words[i - 1] not in [327, 109, 50, 562]:
                if [words[i - 1]] + candidate in names:
                    continue
                return i
    return -1


def segmentation_words(words, sections):
    section_indices = {}
    for section, names in sections.items():
        for name_index, name_words in enumerate(names):
            index = find_index(words, name_words, names[:name_index])
            if index != -1:
                section_indices[section] = (index, index + len(name_words))
                break

    section_indices = sorted(section_indices.items(), key=lambda e: e[1][0])
    section_indices.append(('', (len(words), 0)))
    note_sections = {}
    for i, (section, (start, end)) in enumerate(section_indices[:-1]):
        next_start = section_indices[i + 1][1][0]
        if next_start < end:
            note_sections[section] = []
        else:
            note_sections[section] = words[end:next_start]
    if len(note_sections) == 0:
        note_sections['others'] = words
    else:
        note_sections['others'] = []
    all_note_sections = {}
    for section in sections:
        all_note_sections[section] = note_sections[section] if section in note_sections else []
    return all_note_sections


def segmentation_dataset(dataset, section_titles, tokenizer, word2id, id2word):
    section_ids = {}
    for section, names in section_titles.items():
        name_ids = []
        for name in names:
            words = tokenizer.tokenize(name)
            word_ids = [word2id[word] for word in words]
            name_ids.append(word_ids)
        section_ids[section] = name_ids

    result = []
    for sample in tqdm(dataset):
        text = sample['text']
        words = [word for word in tokenizer.tokenize(text.lower()) if not word.isnumeric()]
        word_ids = [word2id.get(word, word2id['**UNK**']) for word in words]

        note_sections = segmentation_words(word_ids, section_ids)
        for section, section_word_ids in note_sections.items():
            note_sections[section] = ' '.join([id2word[word_id] for word_id in section_word_ids])

        note = {
            'hadm_id': sample['hadm_id'],
            'labels': sample['labels'],
            'sections': note_sections,
            'text': text
        }
        result.append(note)
    return result


def load_vocab(path):
    model = gensim.models.Word2Vec.load(path)
    words = list(model.wv.key_to_index)
    del model

    with open(os.path.join(MIMIC3_PATH, 'word_count_dict.json'), 'r') as f:
        word_count_dict = json.load(f)
    words = [w for w in words if w in word_count_dict]

    for w in ['**UNK**', '**PAD**', '**MASK**']:
        if not w in words:
            words = words + [w]
    word2id = {word: idx for idx, word in enumerate(words)}
    id2word = {idx: word for idx, word in enumerate(words)}
    return word2id, id2word


if __name__ == '__main__':
    if not os.path.exists(SECTION_PATH) or not os.path.isdir(SECTION_PATH):
        os.makedirs(SECTION_PATH)
    section_titles = json.load(open(os.path.join(TITLE_PATH, 'title_synonyms.json')))
    print(section_titles)

    tokenizer = RegexpTokenizer(r'\w+')
    word2id, id2word = load_vocab(os.path.join(MIMIC3_PATH, 'word2vec_sg0_100.model'))

    for task_name in ['mimic3', 'mimic3-50', 'mimic3-50l']:
        for version in ['train', 'dev', 'test']:
            print(f'{task_name}_{version}')
            dataset = json.load(open(os.path.join(EXTRACTED_PATH, f'{task_name}_{version}.json'), encoding='utf-8'))
            new_dataset = segmentation_dataset(dataset, section_titles, tokenizer, word2id, id2word)
            json.dump(new_dataset, open(os.path.join(SECTION_PATH, f'{task_name}_{version}.json'), 'w', encoding='utf-8'), indent=4)
