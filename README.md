# STSM

本项目是论文《Spatio-temporal stacking model for skeleton-based action recognition》的代码实现。

## 安装

这个项目使用anaconda作为包管理器，使用pycharm作为IDE，基于 [numpy1.19.4](https://pypi.org/project/numpy/1.19.4/) 和 [sklearn0.23.2](https://scikit-learn.org/stable/whats_new/v0.23.html)进行开发。请确保本地已安装这些依赖库。更多依赖库，请查看requirements.txt文件。

```sh
conda create -n stsm python=3.7
conda activate stsm
pip install -r requirements.txt
```

## 数据集

数据集存放在[百度云盘](https://pan.baidu.com/s/1e9zWHZ7J4Nyw7p9eOP7Ckw?pwd=amaa)，提取码: amaa。下载后放到STSM文件夹下，并重命名为dataset。

## 使用说明

### 特征提取

我们将数据集的每个动作文件，视作一个样本；将样本数据处理为（骨骼点数量，帧数量）大小的矩阵；计算矩阵的SPD矩阵作为样本的空间特征。

本项目通过data/dataset/*.py文件进行特征提取，现以data/dataset/utd.py为例进行简单介绍。

```python
def utd(root_path, rerange, para, save, window, overlap):
"""
root_path：str型，表示文件夹根目录
rerange：bool型，True表示对骨骼点进行重新排序
para：list型，输入为核函数
save：bool型，True表示对结果进行保存
window：int型，表示时间分片的等级
overlap：bool型，True表示对前一个时间窗口进行重叠
"""

# 第63行
#'linger'表示线性核，'-1'无含义，只是为了保持参数个数的统一
res = utd(root_path, False, ['linger', '-1'], False, 2, True)
#'cov'表示协方差核，'-1'无含义，只是为了保持参数个数的统一
res = utd(root_path, False, ['cov', '-1'], False, 2, True)
#'rbf'表示高斯核，'400'表示高斯核的标准差
res = utd(root_path, False, ['rbf', '400'], False, 2, True)
#'laplace'表示拉普拉斯核，'1e-4'表示拉普拉斯核的方差倒数
res = utd(root_path, False, ['laplace', '1e-4'], False, 2, True)
#'poly'表示多项式核，'1'表示多项式核的次数
res = utd(root_path, False, ['poly', '1'], False, 2, True)
#'sigmoid'表示sigmoid核，'5e-5'表示sigmoid核的坡度(slope)
res = utd(root_path, False, ['sigmoid', '5e-5'], False, 2, True)
```

本项目包含机器学习中常用的一些核函数：线性核、协方差核、高斯核、拉普拉斯核、多项式核、sigmoid核。核函数的具体介绍可参考[Kernels](https://scikit-learn.org/0.23/modules/metrics.html#metrics)。

本项目使用时间分片方法，提取骨骼点的时间特征。具体而言，参数window表示时间分片的等级，当window=1时，时间分片等级为1，即只有一层，实际上不进行操作；当window=2时，时间分片等级为2，即有两层，第二层的矩阵行向量为第一层矩阵行向量的一半。参数overlap表示时间窗口的重叠关系，当overlap=True时，同一层的时间窗口，后一个时间窗口可以覆盖前一个时间窗口的一部分；当overlap=False时，同一层的时间窗口之间无交集。

然而，选择如何选择核函数更好，或者如何选择核函数的参数更好，是一个依赖于经验的问题。可以单独运行utd.py文件，设置第四个参数save为True，保存结果为txt文件。观察txt文件，如果大量元素相同或趋紧于某一个数（如0，1），那么这个核函数的参数是不太好的；应该根据核函数的特点，调整参数，使元素均匀分布。当然更直接的办法，是在分类器中进行测试，大概需要十多分钟左右的时间。

总而言之，先通过保存结果为txt文件，可以很快缩小参数的范围；然后对比分类器的运行结果，可以得到最佳的参数。

### 模型训练

#### 分类算法

本项目测试了sklearn中常见的分类算法，包括[逻辑回归](https://scikit-learn.org/0.23/modules/linear_model.html#logistic-regression)和[支持向量机](https://scikit-learn.org/0.23/modules/svm.html#svc)。对于每一个分类算法，使用[网格搜索](http://scikit-learn.org/0.23/modules/generated/sklearn.model_selection.GridSearchCV.html)寻找最优参数。以上两个算法具体实现可查看models/svm.py和models/logisticregression.py。

#### 集成算法

本项目采用了mlxtend实现的集成算法[stacking](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/)，具体实现可查看models/ensemble_stacking.py。

## 参数列表

本项目经过实验，得出下列最优参数。

| 数据集 | 核函数                 | 时间分片      | 分类器1                      | 分类器2                        | 元分类器                      |
|:--- |:-------------------:| ---------:| ------------------------- | --------------------------- | ------------------------- |
| flo | ['laplace', '1e-4'] | 2+overlap | LogisticRegression(C=1e1) | SVC(C=1e4, kernel='linear') | LogisticRegression(C=1e1) |
| utk | ['laplace', '3e-1'] | 2+overlap | LogisticRegression(C=1e1) | SVC(C=1e4, kernel='linear') | LogisticRegression(C=1e1) |
| msr | ['laplace', '5e-2'] | 2+overlap | LogisticRegression(C=1e3) | SVC(C=1e4, kernel='linear') | LogisticRegression(C=1e5) |
| utd | ['laplace', '1e-1'] | 2+overlap | LogisticRegression(C=1e3) | SVC(C=1e4, kernel='linear') | LogisticRegression(C=1e3) |
| g3d | ['laplace', '5e-1'] | 5+overlap | LogisticRegression(C=1e1) | SVC(C=1e4, kernel='linear') | LogisticRegression(C=1e3) |
| hdm | ['laplace', '5e-4'] | 4+overlap | LogisticRegression(C=1e1) | SVC(C=1e4, kernel='linear') | LogisticRegression(C=5e5) |