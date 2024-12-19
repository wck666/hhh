import logging

from bert_trainer import Trainer
from confs import Config

logger = logging.getLogger(__name__)

import os
# 训练数据数量
# 定义数据集
from paddlenlp.datasets import MapDataset
from preprocess import MyDataset, gen_data_load
# from run_bert_409593 import gen_data_load, QQRProcessor

if __name__ == '__main__':
    label2id = {"0": 0, "1": 1, "2": 2}
    id2label = {0: "0", 1: "1", 2: "2"}
    config = Config()
    config.num_tags = len(label2id.keys())
    print(config)

    train_flag = True
    test_flag = True
    predict_flag = False


    # initialize tokenizer
    tokenizer = config.tokenizer
    # 训练和验证
    print('initialize dataloader')
    train_data = MyDataset()
    test_data = MyDataset('test')
    dev_data = MyDataset('dev')

    # 转换为paddleNPL专用的数据集格式
    train_data = MapDataset(train_data)
    test_data = MapDataset(test_data)
    dev_data = MapDataset(dev_data)
    print("gen_data_load start...")
    train_loader = gen_data_load(train_data, config.tokenizer, config.batch_size, config.max_length, shuffle=True)
    dev_loader = gen_data_load(dev_data, config.tokenizer, config.batch_size, config.max_length, shuffle=False)
    test_loader = gen_data_load(test_data, config.tokenizer, config.batch_size, config.max_length, shuffle=False)

    if train_flag:
        print("train...")
        trainer = Trainer(config, train_loader, dev_loader, dev_loader)
        trainer.train()
    if test_flag:
        # 测试
        print('========进行测试========')
        trainer = Trainer(config, None, None, test_loader)
        checkpoint_path = os.path.join(config.output_dir, 'best.pt')
        trainer.test(checkpoint_path)
