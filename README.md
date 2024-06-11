# Ruoyi_recognize_captcha_model
基于读光快速训练的若依验证码检测模型


## 1. 项目简介
本项目是基于魔塔社区读光ocr通用模型快速训练的若依验证码检测模型，用于识别若依验证码。

## 2. 项目结构
```
├── captcha_recognize_model # 识别模型封装模块
├── config # 配置文件
├── datasets # 采集图片数据集
├── dbset_build.py # 构建lmdb数据集
├── export_onnx_model.py # 导出onnx模型
├── img # 图片下载
├── lmdb_dir # lmdb数据集
├── runs  # 训练日志
├── test.py # 测试代码
└── train.py # 训练代码
```


## 3. 项目使用
### 3.1 数据集准备
通过img下jar包生成图片数据集，然后通过dbset_build.py构建lmdb数据集。

### 3.2 模型训练
运行train.py进行模型训练。 

### 3.3 模型导出
运行export_onnx_model.py导出onnx模型。

### 3.4 模型测试
运行test.py进行模型测试。

## 4. 模型效果
![img.png](img.png)
