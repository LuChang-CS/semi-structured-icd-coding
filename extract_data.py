from typing import Dict, List
import json
import os

import pandas as pd

from conf import IDS_PATH, MIMIC3_PATH, EXTRACTED_PATH


def to_standard_icd9(code: str) -> str:
    split_pos = 4 if code.startswith('E') else 3
    icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
    return icd9_code


def read_ids(path: str) -> List[str]:
    with open(path, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    return ids


def read_note(path: str) -> Dict[str, str]:
    df = pd.read_csv(path,
                     usecols=['HADM_ID', 'CATEGORY', 'TEXT'],
                     converters={'HADM_ID': str})
    notes = {row.HADM_ID: row.TEXT for row in df.itertuples(index=False) if row.CATEGORY == 'Discharge summary'}
    return notes


def read_diagnoses(path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path,
                     usecols=['HADM_ID', 'ICD9_CODE'],
                     converters={'HADM_ID': str, 'ICD9_CODE': str})
    diagnoses = {}
    for row in df.itertuples(index=False):
        hadm_id, code = row.HADM_ID, to_standard_icd9(row.ICD9_CODE)
        if hadm_id in diagnoses:
            diagnoses[hadm_id].add(code)
        else:
            diagnoses[hadm_id] = {code}
    return diagnoses


if __name__ == '__main__':
    note_path = os.path.join(MIMIC3_PATH, 'NOTEEVENTS.csv')
    diagnoses_path = os.path.join(MIMIC3_PATH, 'DIAGNOSES_ICD.csv')
    if not os.path.exists(EXTRACTED_PATH) or not os.path.isdir(EXTRACTED_PATH):
        os.makedirs(EXTRACTED_PATH)

    all_notes, all_diagnoses = read_note(note_path), read_diagnoses(diagnoses_path)

    for task_name in ('mimic3', 'mimic3-50', 'mimic3-50l'):
        for version in ('train', 'dev', 'test'):
            print(f'processing {task_name}_{version}')
            ids = read_ids(os.path.join(IDS_PATH, f'{task_name}_{version}_ids.txt'))
            data = [
                {
                    'hadm_id': hadm_id,
                    'text': all_notes[hadm_id],
                    'labels': list(all_diagnoses[hadm_id])
                }
                for hadm_id in ids
            ]
            json.dump(
                data,
                open(os.path.join(EXTRACTED_PATH, f'{task_name}_{version}.json'), 'w', encoding='utf-8'),
                indent=4
            )
