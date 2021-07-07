[comment]: <> (models.py)
    models.py
****
# class NetModel
**管理模型配置文件(.cfg)上传、下载、删除等，模型文件后缀是 .cfg 文件**

nm_number ： 神经网络模型的编号（唯一）\
nm_name ： 神经网络模型的名称（版本号）\
nm_path ： 神经网络模型保存的静态地址 \
nm_time ： 神经网络模型上传时间

****
# class Checkpoint
**管理每个网络模型训练获得的模型文件(.pt/pth)，文件后缀 .pth/.pt**

ck_number ： 模型 checkpoint 编号（唯一）\
ck_time ： 模型 checkpoint 保存的时间 \
ck_path ： 模型 checkpoint 保存的静态地址 \
ck_model ： 外健，对应模型

****
