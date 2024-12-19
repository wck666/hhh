# 分词，batch化
import functools
import numpy as np

from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding

# 定义数据集
from paddlenlp.datasets import MapDataset

from PIL import Image
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import pickle
from visualdl import LogWriter
from tqdm import tqdm
import json
import copy


class MyDataset(paddle.io.Dataset):
    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        self.mode = mode

        if mode == 'train':
            self.data_path = 'data/KUAKE-QQR_train.json'
        if mode == 'dev':
            self.data_path = 'data/KUAKE-QQR_dev.json'
        if mode == 'test':
            self.data_path = 'data/KUAKE-QQR_test.json'

        with open(self.data_path, 'r', encoding='utf-8') as input_data:
            self.json_content = json.load(input_data)
            self.data = []
            for block in self.json_content:
                sample = {}
                sample['record_id'] = block['id']
                sample['query1'] = block['query1']
                sample['query2'] = block['query2']
                sample['label'] = block['label']
                if sample['label'] == 'NA':
                    continue
                self.data.append(sample)

    def __getitem__(self, idx):
        query1 = self.data[idx]['query1']
        query2 = self.data[idx]['query2']
        label = self.data[idx]['label']

        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def label_num(self):
        # 返回该数据集下各个label的个数
        label_0, label_1, label_2 = 0, 0, 0
        for sample in self.data:
            if sample['label'] == '0':
                label_0 += 1
            if sample['label'] == '1':
                label_1 += 1
            if sample['label'] == '2':
                label_2 += 1
        return (label_0, label_1, label_2)


# 数据预处理函数，利用分词器将文本转化为整数序列
def preprocess_function(examples, tokenizer, max_seq_length, is_test=False):
    result = tokenizer(text=examples["query1"], text_pair=examples["query2"], max_length=max_seq_length)
    if not is_test:
        if examples["label"] == '0':
            result["labels"] = 0
        elif examples["label"] == '1':
            result["labels"] = 1
        elif examples["label"] == '2':
            result["labels"] = 2
        else:
            print(examples)

        # result["labels"] = int(examples["label"])
    else:
        result['id'] = int(examples["record_id"][1:])  # record_id = 's1'
    return result


def gen_data_load(train_data: MapDataset,  tokenizer, batch_size: int, max_length:int, shuffle=False):
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=max_length)
    train_data_cp = copy.deepcopy(train_data).map(trans_func)

    print(train_data_cp[0])
    # collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
    collate_fn = DataCollatorWithPadding(tokenizer, padding=True)

    # 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader

    train_batch_sampler = BatchSampler(train_data_cp, batch_size=batch_size, shuffle=shuffle)

    # 数据集定义
    train_data_loader = DataLoader(dataset=train_data_cp, batch_sampler=train_batch_sampler,
                                   collate_fn=collate_fn)

    for _, data in enumerate(train_data_loader, start=1):
        print("gen_data_load:pp")
        print(data)
        break
    print("gen_data_load end...")
    return train_data_loader