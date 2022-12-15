baseline原文地址：https://discussion.datafountain.cn/articles/detail/2513

BERT模型：https://huggingface.co/hfl/chinese-roberta-wwm-ext, 下载pytorch_model.bin放到hfl文件夹下

类别数量在model.py line13: class_num = 32, 根据需要更改。

训练数据：data/new_train.json, new_train_aug_trnas.json为中英回译结果（第一行原文本，第二行回译结果）；new_train_aug_trnas_eda.json在基础上每条增加3个数据增强结果（第1-4行：原文本及增强结果，5-8行：回译文本及增强结果）。

data_helper.py line21：简单的文本复制增强，按需尝试。

训练：train.py
测试：test.py; 在config.py中更改加载的模型权重路径

主要包：pytorch, transformers