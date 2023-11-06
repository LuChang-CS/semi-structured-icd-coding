import os
import json
import pickle

from zss import simple_distance
from pptree import print_tree, Node
from tqdm import tqdm

from conf import DATA_PATH, EXTRACTED_PATH


class ICDNode(Node):
    def __init__(self, name, children=None):
        super().__init__(name, parent=None)
        self.label = name
        self.children = [] if children is None else children

    def get_children(self):
        return self.children

    def get_label(self):
        return self.label

    def add_child(self, child):
        self.children.append(child)


def build_icd_tree():
    def build(code):
        node = ICDNode(code['code'])
        for child in code['children']:
            node.add_child(build(child))
        return node

    root = ICDNode('#')
    icd_codes = json.load(open(os.path.join(DATA_PATH, 'diagnosis_codes.json'), 'r', encoding='utf-8'))
    for code in icd_codes:
        root.add_child(build(code))
    return root


def build_label_tree(icd_root, labels):
    labels = set(labels)
    all_nodes = set()

    def build(node):
        result_children = []
        for child in node.children:
            result_child = build(child)
            if result_child is not None:
                result_children.append(result_child)
        if node.label in labels or len(result_children) > 0:
            all_nodes.add(node.label)
            return ICDNode(node.label, children=result_children)
        return None

    result_root = build(icd_root)
    return result_root, all_nodes


def build_label_trees():
    cached_path = os.path.join(EXTRACTED_PATH, 'label_trees.pkl')
    if os.path.exists(cached_path) and os.path.isfile(cached_path):
        label_trees = pickle.load(open(cached_path, 'rb'))
        return label_trees
    icd_root = build_icd_tree()
    train_data = json.load(open(os.path.join(EXTRACTED_PATH, 'mimic3_train.json'), 'r', encoding='utf-8'))
    label_trees = {}
    for sample in tqdm(train_data):
        hadm_id, labels = sample['hadm_id'], sample['labels']
        label_tree, all_nodes = build_label_tree(icd_root, labels)
        label_trees[hadm_id] = (label_tree, all_nodes)
    pickle.dump(label_trees, open(cached_path, 'wb'))
    return label_trees


def calc_label_similarity(label_tree_1, all_nodes_1, label_tree_2, all_nodes_2):
    dist = simple_distance(label_tree_1, label_tree_2)
    similarity = 1 - 2 * dist / (len(all_nodes_1 | all_nodes_2) - 1)
    return similarity


if __name__ == '__main__':
    icd_root = build_icd_tree()

    labels_1 = ['401.9', '596.54', '788.30', 'V10.46', '493.90', '410.71', '285.1', '599.70', '362.50', '272.4', '564.1', '578.0']
    labels_2 = ['410.41', '250.00', '401.9', '414.01']
    label_tree_1, all_nodes_1 = build_label_tree(icd_root, labels_1)
    label_tree_2, all_nodes_2 = build_label_tree(icd_root, labels_2)
    print_tree(label_tree_1)
    print_tree(label_tree_2)
    print('similarity:', calc_label_similarity(label_tree_1, all_nodes_1, label_tree_2, all_nodes_2))
