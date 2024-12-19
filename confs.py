import argparse
import os
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

# from transformers import BertTokenizer,AutoTokenizer

# 获取当前脚本所在的目录
current_path = os.path.dirname(os.path.abspath(__file__))

# 获取当前脚本所在的项目根目录
root_path = os.path.dirname(current_path)

print("项目根目录路径：", root_path)

run_env = os.getenv('ENV', 'NULL')

print("当前环境：", run_env)

model_root_dir = '/data/nfs/baozhi/models'

data_root_dir = 'data/'


class Config(object):
    """配置参数"""

    def __init__(self):
        self.hidden_size = 768
        self.batch_size = 64

        # 训练时使用

        # ***** epochs轮数 *****
        self.train_epochs = 10
        self.seed = 123
        self.lr = 5e-5
        self.other_lr = 3e-4
        self.early_stop = 20  # 验证多少次，loss没有上升，提前结束训练

        # 超参数设置
        self.log_steps = 10 * int(128 / self.batch_size)  # 多少个step展示一次训练结果
        self.eval_steps = 10 * int(128 / self.batch_size)  # 每隔几步评估模型，同时保存模型

        # 分类标签数量
        self.num_tags = 3
        self.dataloader_num_workers = 4
        self.max_length = 128
        self.dropout = 0.1

        # 如果不存在该文件，需要解压缩company.rar文件

        self.data_dir = data_root_dir
        self.log_dir = data_root_dir + "logs/"
        self.output_dir = data_root_dir + "/output_data/"
        self.output_redict = data_root_dir + "/output_data/predict.json"

        # 本地环境
        if run_env == 'local':
            # self.model_path = "E:/models/ernie-health-zh"
            # self.model_path = "E:/models/chinese-bert-wwm-ext"
            # self.model_path = "E:/models/bert-base-chinese"
            # 如果本地没有直接使用：ernie-health-chinese，会自动从网上下载
            # self.model_path = "E:/models/ernie-health-chinese"
            self.model_path = "ernie-3.0-xbase-zh"
        else:
            # self.model_path = '/data/nfs/baozhi/models/ernie-health-zh'
            # self.model_path = "/data/nfs/baozhi/models/ernie-3.0-base-zh"
            # self.model_path = '/data/nfs/baozhi/models/chinese-bert-wwm-ext'
            # self.model_path = "/data/nfs/baozhi/models/google-bert_bert-base-chinese"
            # 如果本地没有直接使用：ernie-health-chinese，会自动从网上下载
            # self.model_path = "/data/nfs/baozhi/models/ernie-health-chinese"
            self.model_path = "ernie-3.0-xbase-zh"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding=True, max_length=self.max_length,
                                                       truncation=True)
        print('confs end...')
        print(self.tokenizer)
