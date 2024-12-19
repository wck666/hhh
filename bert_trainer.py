import logging

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from eval import evaluate

logger = logging.getLogger(__name__)
import torch.nn.functional as F
import paddle
import os
from visualdl import LogWriter
import json
from confs import Config
# 开始训练
import time
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModelForSequenceClassification


# 根据概率排序
def sort_list(data_list, min=0.5, size=3):
    ids = []
    data_dict = {}
    if not data_list or len(data_list) < 1:
        return ids
    for index, item in enumerate(data_list[0]):
        if item and type(item) == float:
            data_dict[index] = item
    sort_result = sorted(data_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for index, value in sort_result:
        if len(ids) >= size:
            break
        if value > min:
            ids.append(index)
    return ids


class Trainer:
    def __init__(self, args: Config, train_loader, dev_loader, test_loader):
        self.checkpoint = None
        self.args = args
        self.device_type = 'cpu'
        # self.device_type2 = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)

        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_classes=args.num_tags, dropout=args.dropout)
        # Adam优化器、交叉熵损失函数、accuracy评价指标
        self.optimizer = paddle.optimizer.AdamW(learning_rate=self.args.lr, parameters=self.model.parameters())
        self.criterion = paddle.nn.loss.CrossEntropyLoss()
        # 设置类别的Loss权重
        self.class_weight = paddle.to_tensor([1.3, 1.0, 1.0], dtype='float32')
        self.criterion = paddle.nn.loss.CrossEntropyLoss(weight=self.class_weight)

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        # self.model.to(self.device)

    def get_params(self):
        checkpoint = self.checkpoint
        return self.model, self.optimizer, checkpoint['epoch'], checkpoint['loss']

    def load_ckp(self, model, optimizer, check_point_path):
        if self.device.type == 'cpu':
            checkpoint = torch.load(check_point_path, map_location=torch.device('cpu'))
            print('checkpoint use cpu')
        else:
            checkpoint = torch.load(check_point_path)
            print('checkpoint use gpu')
        self.checkpoint = checkpoint

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return model, optimizer, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    """
    def save_ckp(self, state, is_best, checkpoint_path, best_model_path):
        tmp_checkpoint_path = checkpoint_path
        torch.save(state, tmp_checkpoint_path)
        if is_best:
            tmp_best_model_path = best_model_path
            shutil.copyfile(tmp_checkpoint_path, tmp_best_model_path)
    """



    # def evaluate( model, class_list, data_iter, test=False):
    #     model.eval()
    #     loss_total = 0
    #     predict_all = np.array([], dtype=int)
    #     labels_all = np.array([], dtype=int)
    #     with torch.no_grad():
    #         for texts, labels in data_iter:
    #             outputs = model(texts)
    #             loss = F.cross_entropy(outputs, labels)
    #             loss_total += loss
    #             labels = labels.data.cpu().numpy()
    #             predic = torch.max(outputs.data, 1)[1].cpu().numpy()
    #             labels_all = np.append(labels_all, labels)
    #             predict_all = np.append(predict_all, predic)
    #     p = metrics.precision_score(labels_all, predict_all, average='macro')
    #     r = metrics.recall_score(labels_all, predict_all, average='macro')
    #     f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    #     acc = metrics.accuracy_score(labels_all, predict_all)
    #     if test:
    #         report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
    #         confusion = metrics.confusion_matrix(labels_all, predict_all)
    #         # return acc, loss_total / len(data_iter), report, confusion, predict_all
    #         return p, r, f1, acc, loss_total / len(data_iter), report, confusion, predict_all
    #     # return acc, loss_total / len(data_iter)
    #     return p, r, f1, acc, loss_total / len(data_iter)

    def train(self):
        # 冻结模型
        # ernie.pooler
        name_list = []
        for name, param in self.model.named_parameters():
            name_list.append(name)
            if not name.startswith('classifier') and not name.startswith('roberta.pooler'):
                param.requires_grad = False
            else:
                param.requires_grad = True

        print(name_list[-10:])

        # 超参数设置
        lr = self.args.lr  # 学习率
        epochs = self.args.train_epochs  # 训练轮次
        early_stop = self.args.early_stop  # 验证多少次，loss没有上升，提前结束训练
        save_dir = self.args.output_dir  # 训练过程中保存模型参数的文件夹
        log_dir = self.args.log_dir  # VisaulDL的保存路径
        twriter = LogWriter(log_dir)

        metric = paddle.metric.Accuracy()
        # rdrop_loss = paddlenlp.losses.RDropLoss()  # R-Drop数据增广

        stop_trainning = False
        best_train_dev_error = 1
        eval_times = 0
        best_loss = 1000.
        best_acc = 0
        best_step = 0
        global_step = 0  # 迭代次数
        tic_train = time.time()

        # 可以看一下整个模型的结构
        print(self.model)
        for epoch in range(self.args.train_epochs):
            if stop_trainning:
                break
            for step, batch in enumerate(self.train_loader):
                self.model.train()
                print(step)
                # input_ids, token_type_ids, labels, attention_mask = batch['input_ids'], batch['token_type_ids'], batch['labels'], batch['attention_mask']
                input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']
                # 计算模型输出、损失函数值、分类概率值、准确率
                logits = self.model(input_ids, token_type_ids)
                # logits = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(logits, labels)
                probs = F.softmax(logits, axis=1)
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()

                # # R-Drop数据增强
                # logits_2 = model(input_ids, token_type_ids)  # 因为dropout层的存在，每个时刻的模型的输出都不同
                # kl_loss = rdrop_loss(logits, logits_2)
                # loss = loss + kl_loss

                # 每迭代10次，打印损失函数值、准确率、计算速度
                global_step += 1
                if global_step % self.args.log_steps == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss, acc,
                           10 / (time.time() - tic_train)))
                    tic_train = time.time()

                # VisualDL 推流
                twriter.add_scalar('loss', loss, global_step)
                twriter.add_scalar('train acc', acc, global_step)

                # 反向梯度回传，更新参数
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

                # 评估当前训练的模型、保存当前最佳模型参数和分词器的词表等
                if global_step % self.args.eval_steps == 0:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    print("global step", global_step, end=' ')
                    acc_eval, loss_eval = evaluate(self.model, self.criterion, metric, self.dev_loader)

                    # VisualDL 推流
                    twriter.add_scalar('eval acc', acc_eval, global_step)
                    twriter.add_scalar('eval loss', loss_eval, global_step)

                    eval_times += 1
                    if eval_times > early_stop:
                        print('-----------------Early Stopping-----------------')
                        stop_trainning = True
                        break

                    # 保存模型
                    if acc_eval > best_acc and loss_eval < best_loss:
                        best_acc = acc_eval
                        best_loss = loss_eval
                        best_step = global_step

                        self.model.save_pretrained(save_dir)
                        self.args.tokenizer.save_pretrained(save_dir)
                        print('save model to {}'.format(save_dir))
                        eval_times = 0

                    # 这里是保存一些，虽然loss和acc没有同时升高，但是train acc和eval acc都比较可观的模型
                    if acc_eval > 0.84 and acc > 0.90:
                        best_train_dev_error = abs(acc - acc_eval)
                        new_save_dir = save_dir[:-1] + '_' + str(global_step) + '/'
                        if not os.path.exists(new_save_dir):
                            os.makedirs(new_save_dir)
                        print('new save model to {}'.format(new_save_dir))
                        self.model.save_pretrained(new_save_dir)
                        self.args.tokenizer.save_pretrained(new_save_dir)

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['input_ids'].to(self.device)
                attention_masks = dev_data['attention_mask'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids)
                self.model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs1 = torch.max(outputs, 1)[1]
                dev_outputs.extend(outputs1.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        optimizer = self.optimizer
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint_path)
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        index = 0
        test_examples = self.test_loader.dataset.examples
        result_examples = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['input_ids'].to(self.device)
                attention_masks = test_data['attention_mask'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)

                outputs = model(token_ids, attention_masks, token_type_ids)
                self.model.zero_grad()
                outputs1 = torch.max(outputs, 1)[1].cpu()
                outputs1 = outputs1.cpu().detach().numpy().tolist()

                test_outputs.extend(outputs1)

                for item in outputs1:
                    example = test_examples[index]
                    example.label = str(item)
                    index += 1
                    result_examples.append({"id": example.guid, "query1":example.text_a, "query2": example.text_b, "label": example.label})

        for example in test_examples:
            result_examples.append(
                {"id": example.guid, "query1": example.text_a, "query2": example.text_b, "label": example.label})
        with open(self.args.output_redict, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_examples, ensure_ascii=False))
        assert len(test_outputs) == len(test_examples)
        print(len(test_examples))

    def predict(self, tokenizer, text, id2label, max_seq_len):
        model, optimizer, epoch, loss = self.get_params()

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=max_seq_len,
                                           truncation='longest_first',
                                           padding="max_length",
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
            token_ids = inputs['input_ids'].to(self.device)
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            outputs = model(token_ids, attention_masks, token_type_ids)
            self.model.zero_grad()

            outputs1 = torch.max(outputs, 1)[1]
            if len(outputs1) != 0:
                return outputs1[0]
            else:
                return None

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1

    def get_classification_report(self, outputs, targets, labels):
        # confusion_matrix = multilabel_confusion_matrix(targets, outputs)
        report = classification_report(targets, outputs, target_names=labels)
        return report
