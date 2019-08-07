# Deep Learning Specialization 学习笔记

吴恩达的Deep Learning Specialization 知识点整理, 内容暂时跳过了CNN部分, 建议与课程一起食用, 这部分作为复习时快速过知识点, 课程参考[Coursera](https://www.coursera.org/specializations/deep-learning). 

Course 1 Neural Networks and Deep Learning

- [ Week2](https://nbviewer.jupyter.org/github/caidwang/NN-and-DL-Notebook/blob/master/C1Week2.ipynb)
    - logistic回归 -- 二分类器
        - sigmoid函数
        - loss函数 && MSE和Cross Entropy方法对比
        - cost函数
        - 前向传播与反向传播
    - numpy中的广播机制
    - numpy的二分类器实现

- [Week3](https://nbviewer.jupyter.org/github/caidwang/NN-and-DL-Notebook/blob/master/C1Week3.ipynb)
    - 浅层神经网络
    - 神经网络的表示
    - 正向传播和反向传播
    - 激活函数 && sigmoid和relu, tanh的比较
    - 梯度下降的学习方法
    - 随机初始化

Course 2 Hyperparameter tuning, Regularization and Optimization

- [Week1](https://nbviewer.jupyter.org/github/caidwang/NN-and-DL-Notebook/blob/master/C2Week1.ipynb)
    - train/Dev/Test Set
    - 利用Bias和Variance诊断模型
    - 正则化
        - L1/L2
        - Dropout
        - others
    - 归一化
    - 梯度爆炸和梯度消失
    - Xavier参数初始化

- [Week2](https://nbviewer.jupyter.org/github/caidwang/NN-and-DL-Notebook/blob/master/C2Week2.ipynb)
    - Mini-batch训练方法和优点
    - 指数加权平均
        - Bias Correction
    - Momentum SGD
    - RMSprop
    - Adam
    - 学习率衰减
        - 神经网络优化中的局部最优问题

- [Week3](https://nbviewer.jupyter.org/github/caidwang/NN-and-DL-Notebook/blob/master/C2Week3.ipynb)
    - 超参调整
        - 调整顺序(重要性排列)
        - 调整方法
    - Batch Normalization
        - 计算方法
        - 机制有效性
        - BN在test过程的正确性
    - TensorFlow基础
        - tensorflow基本操作
        - 使用tensorFlow训练全连接多分类器

Course3 Structuring Machine Learning Projects **Todo!**

Course5 Sequence Model

- [Week1](https://nbviewer.jupyter.org/github/caidwang/NN-and-DL-Notebook/blob/master/C5Week1.ipynb)
    - RNN 基本结构
      - RNN的正向传播
      - BPTT反向传播
      - 梯度消失问题
      - 梯度爆炸/Clipping方法
      - RNN的numpy实现
    - GRU
        - pytorch实现
    - LSTM
        - pytorch实现
    - 双向RNN
    - Deep RNN
- [Week2](https://nbviewer.jupyter.org/github/caidwang/NN-and-DL-Notebook/blob/master/C5Week2.ipynb) **!todo**
    - 词嵌入
    - Word2Vec && GloVe
- [Week3](https://nbviewer.jupyter.org/github/caidwang/NN-and-DL-Notebook/blob/master/C5Week3.ipynb)
    - Seq2Seq
    - Beam Search
    - Attention Model
    - Attention变种
        - Self Attention
        - Multi-head Attention

Others

- [如何在coursera上申请奖学金免费学习完整课程](https://github.com/caidwang/NN-and-DL-Notebook/blob/master/How%20to%20Apply%20Financial%20Aid%20for%20this%20Specialization.md)
- [pytorch快速入门](https://blog.csdn.net/m0_37407587/article/details/96479154)
