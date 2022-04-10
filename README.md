# 神经网络与深度学习课程
2022春季学期神经网络与深度学习课程作业1：不使用Pytorch等现有库实现MNIST数据集的两层神经网络分类器

## 文件说明
### data.zip 
MNIST数据集
### functions.py
定义的一些函数，包括解析数据文件的函数，以及实现对数据集调用训练好模型功能的函数
### main.py
定义了分类器的类(Sol)，方法包括 训练、反向传播、保存模型、预测、可视化等，以及网格搜索的函数，最终选取模型训练过程的可视化以及参数保存等。

## 训练步骤
1. main.py中获取训练集/测试集
2. main.py中网格搜索找到最优的超参组合（学习率，正则化强度，隐含层大小）记为 lr, regu, hsize
3. 通过2找到的最优超参组合, 创建分类器类的实例 a = Sol(traindata, lr, regu, hsize, False)
4. 通过train方法训练n个epoch： a.train(n)
5. 储存a的最终网络参数为bestpara.pkl, 过程参数（网络参数和Loss等）为processparameters.pkl并可视化训练过程 
（3 、4 和 5 通过调用finalmodel(lr, regu,hsize)函数实现）

## 测试步骤
对想要测试的数据集testdata和对应的标签集testlabel调用predictbymodel(testdata, testlabel)即可.
