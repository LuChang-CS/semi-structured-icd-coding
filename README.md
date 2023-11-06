# Semi Structured ICD Coding

Partial code for the NeurIPS 2023 paper: [Towards Semi-Structured Automatic ICD Coding via Tree-based Contrastive Learning](https://neurips.cc/virtual/2023/poster/71400).

```bibtex
@article{lu2023towards,
  title={Towards Semi-Structured Automatic ICD Coding via Tree-based Contrastive Learning},
  author={Lu, Chang and Reddy, Chandan K and Wang, Ping and Ning, Yue},
  journal={37th Conference on Neural Information Processing Systems (NeurIPS 2023)},
  year={2023}
}
```

## Requirements

```bash
# For CUDA user
pip install -r requirements.txt

# For CPU user, change the torch==1.13.1+cu116 in the requirements.txt to torch==1.13.1
```

## Running

### Data Preprocessing

1. Put `DIAGNOSES_ICD.csv` and `NOTEEVENTS.csv` to `data/mimic3`.
2. Extract json files using `hadm_id` in the `data/mimic3/ids`:
```bash
python extract_data.py
```
The `mimic3` and `mimic3-50` ids are following the [ICD-MSMN](https://github.com/GanjinZero/ICD-MSMN) project. The `mimic3-50l` ids are following the [KEPT](https://github.com/whaleloops/KEPT) project.

3. Extract section titles:
```bash
python df_iapf.py
```
The top 23 extracted titles will be saved to `data/mimic3/titles/titles.txt`. All title candidates will be saved to `data/mimic3/titles/all_titles.txt`.

*Note*: Some optimization based on the paper:
- We only selected sentences containing less than 5 words.
- The final order of the extracted titles may be different from the paper.

*Optional*: Here, since it is strict to require all clinical notes to have the same titles, we performed a title synonym extraction based on cosine similarity between top titles and all titles using [Sentence-Transformer](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). We provide the synonyms in `data/mimic3/titles/title_synonyms.json`.

4. Segment clinical notes into sections:
```bash
python segment.py
```
The sections will be saved to `data/mimic3/sections`.

## Contrastive Pretraining

We provide a dataset class `ContrastiveDataset` in the `dataset.py`. It can be used as follows:
```python
dataset = ContrastiveDataset()

texts, similarity, loss_weight = dataset[0]
# texts: a dict containing positive pairs from s_i and neighbor pairs from s_j:
# texts = {
#     'text_1_i': 'xxx',
#     'text_2_i': 'xxx' or '',
#     'text_1_j': 'xxx' or '',
#     'text_2_j': 'xxx' or ''
# }
# Some field can be an empty string if the clinical note does not have the corresponding part.
#
# similarity: float. The similarity between two label trees.
#
# loss_weight: The indicator to control whether this loss should be reflected in the final loss function.
# loss_weight = {
#     'w_1i_2i': 1 if len(texts['text_1_i']) > 0 and len(texts['text_2_i']) > 0 else 0,
#     'w_1i_1j': 1 if len(texts['text_1_i']) > 0 and len(texts['text_1_j']) > 0 else 0,
#     'w_1j_2j': 1 if len(texts['text_1_j']) > 0 and len(texts['text_2_j']) > 0 else 0,
#     'w_2i_2j': 1 if len(texts['text_2_i']) > 0 and len(texts['text_2_j']) > 0 else 0
# }
```

## Masked Section Training

We provide a dataset class `MaskedSectionDataset` in the `dataset.py`. It can be used as follows:
```python
dataset = MaskedSectionDataset(task='mimic3', version='train', mask_rate=0.2)
# task can be chosen from ['mimic3', 'mimic3-50', 'mimic3-50l'].
# version can be chosen from ['train', 'dev', 'test'].
# mask_rate: float. It is similar to dropout rate.

text = dataset[0]
# When version == 'train', the text will be masked based on section and re-concatenated. Otherwise, the text will be the same.
```

## Acknowledgement

- The `mimic3` and `mimic3-50` ids are following the [ICD-MSMN](https://github.com/GanjinZero/ICD-MSMN) project.
- The `mimic3-50l` ids are following the [KEPT](https://github.com/whaleloops/KEPT) project.
- The `word_count_dict.json` is provided by the [ICD-MSMN](https://github.com/GanjinZero/ICD-MSMN) project.
- The `word2vec` models are provided by the [LAAT](https://github.com/aehrc/LAAT) project.
- The `diagnosis_codes.json` is provided by the [icd_hierarchical_structure](https://github.com/LuChang-CS/icd_hierarchical_structure) project.
