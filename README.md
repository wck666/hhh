
比赛地址：【NLP】医学搜索Query相关性判断
https://tianchi.aliyun.com/competition/entrance/532001/introduction?spm=a2c22.12281925.0.0.684a71371ZOgK4

V1：
长期赛：
分数:0.8258
排名：长期赛:272/23398

方案：BERT
预训练模型:bert-base-chinese

训练结果：
Val P: 0.758,  Val R: 76.6730%,  Val F1: 0.7595,  Val Acc: 81.7135%,  Time: 0:01:28

V2：
长期赛：
分数:0.8534
排名：长期赛:148（本次）/ 23398（团体或个人）

方案：BERT
预训练模型:ernie-health-chinese

训练结果：
loss: 0.50159, accu: 0.82927

V3:使用V2的代码，只要在参数中换个模型名称：ernie-3.0-xbase-zh
长期赛：
分数:0.8703
排名：长期赛:60（本次）/23398（团体或个人）

方案：BERT
预训练模型:ernie-3.0-xbase-zh

训练结果：
？

运行脚本：
python train_main.py


# 安装paddlenlp
#windows版本paddlenlp==2.6.0
python -m pip install paddlenlp==2.6.0 -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

# 安装paddle-gpu
#windows版本paddlepaddle==2.6.1
python -m pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
#python -m pip install paddlepaddle-gpu==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
