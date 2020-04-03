# 简单分类
使用sklearn的tree、贝叶斯、KNN进行简单的分类。
## 训练集拆分
```python
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data,target,test_size=0.2)
```
## 决策树
```python
clf = DecisionTreeClassifier()
clf.fit(xtrain, ytrain)
train_accuracy = clf.score(xtrain, ytrain)
test_accuracy = clf.score(xtest, ytest)
```
|参数|值|说明|
|---|---|---|
|criterion|"entropy"/"gini"|使用基尼系数或者信息熵计算信息增益，一般使用默认的基尼系数。
|random_state|int|随机数种子
|splitter|"best"/"random"|优先选择最重要的分支进行分裂或更加随机地进行分裂。用于防止过拟合

剪枝参数：
|参数|值|说明|
|---|---|---|
|max_depth|int|限制树的最大深度，建议从3开始视过拟合状况而定。
|min_samples_leaf|int|分支后，每个**叶**节点都至少包含多少个训练样本，否则不发生分支，建议从5开始，1通常是最优。搭配max_depth在回归树中表现良好。
|min_samples_split|int|一个节点至少包含多少个训练样本才能分裂。
|max_features|int|限制考虑的特征个数，一般使用max_depth，不用这个。
|min_impurity_decrease|int|限制信息增益的大小，小于阈值分支不发生。

权重参数：
|参数|值|说明|
|---|---|---|
|class_weight|list of dict {label:weight}|对于每个类别按顺序传入一个权重字典
|min_weight_fraction_leaf|float|能够分裂的最小权重

feature_importances_属性用于查看各个特征对模型的重要性

## 朴素贝叶斯
若一个样本有n个特征，分别用x1,x2,…,xn表示，将其划分到类yk的可能性P(yk|x1,x2,…,xn)为：
`P(yk|x1,x2,…,xn)=P(yk)∏ni=1P(xi|yk)`
### 高斯模型
`GaussianNB` 实现了运用于分类的高斯朴素贝叶斯算法。高斯模型假设这些一个特征的所有属于某个类别的观测值符合高斯分布，一般用于特征为连续值的分类（身高、体重等），可能通过正则化使得属性逼近正态分布效果更好。
### 多项式模型
`MultinomialNB`。
该模型常用于文本分类，特征是单词，值是单词的出现次数。类似的数据模型可能都可以用多项式模型来做。

值得注意的是，多项式模型在训练一个数据集结束后可以继续训练其他数据集而无需将两个数据集放在一起进行训练。在sklearn中，MultinomialNB()类的partial_fit()方法可以进行这种训练。这种方式特别适合于训练集大到内存无法一次性放入的情况。

在第一次调用partial_fit()时需要给出所有的分类标号。
### 伯努利模型
`BernoulliNB`在伯努利模型中对于一个样本来说，其特征用的是全局的特征；每个特征的取值是布尔型的，即true和false，或者1和0。
BernoulliNB()类也有partial_fit()函数。

### 参数
`alpha`:先验平滑因子，默认等于1，当等于1时表示拉普拉斯平滑。

`fit_prior`:是否去学习类的先验概率，默认是True

`class_prior`:各个类别的先验概率，如果没有指定，则模型会根据数据自动学习， 每个类别的先验概率相同，等于类标记总个数N分之一。

## KNN
`sklearn.neighbours`
### 参数
+ n_neighbors 就是 kNN 里的 k。

+ weights 是在进行分类判断时给最近邻附上的加权，默认的'uniform' 是等权加权，还有'distance' 选项是按照距离的倒数进行加权，也可以使用用户自己设置的其他加权方法。

+ algorithm 是分类时采取的算法，有'brute'、'kd_tree' 和'ball_tree'。kd_tree 的算法为 kd 树，而 ball_tree 是另一种基于树状结构的 kNN 算法，brute 则是最直接的蛮力计算。根据样本量的大小和特征的维度数量，不同的算法有各自的优势。默认的'auto' 选项会在学习时自动选择最合适的算法，所以一般来讲选择 auto 就可以。

+ leaf_size 是 kd_tree 或 ball_tree 生成的树的树叶（树叶就是二叉树中没有分枝的节点）的大小。对于很多使用场景来说，叶子的大小并不是很重要，我们设leaf_size=1 就好。

+ metric 和 p，距离函数的选项，一般来讲，默认的 metric='minkowski'（默认）和 p=2（默认）就可以满足大部分需求。其他的 metric 用到再查文档

+ metric_params 是一些特殊 metric 选项需要的特定参数，默认是 None。

+ n_jobs 是并行计算的线程数量，默认是 1，输入 -1 则设为 CPU 的内核数。

## 分类器的测度
![](https://lh3.googleusercontent.com/-zydpCRsTS74/Uq1AVlpHNRI/AAAAAAAAAYg/ODL2Uf2WUdg/s0/9686a1f19149fe16eb4b6b383904d086.png)
+ TPR: 真阳性率，所有阳性样本中 (TP+FN)，被分类器正确判断为阳的比例。
    >TPR = TP / (TP + FN) = TP / 所有真实值为阳性的样本个数
+ FPR: 伪阳性率，所有阴性样本中 (FP+TN)，被分类器错误判断为阳的比例。
    >FPR = FP / (FP + TN) = FP / 所有真实值为阴性的样本个数

ROC 空间是一个以伪阳性率 (FPR, false positive rate) 为 X 轴，真阳性率 (TPR, true positive rate) 为 Y 轴的二维坐标系所代表平面。
![](https://lh5.googleusercontent.com/-J8CTsDzjWo8/Uq1D_ZbypHI/AAAAAAAAAYs/vghcF3Ehvy4/s0/1a02adedd70816dcd49461354390aaed.png)
ROC空间的横坐标和纵坐标没有联系，表示一个分类器的性能。左上最好，右下最差，右下半三角反预测而行就是左上半三角，所以不存在右下半三角的分类器，最差是对角线。

ROC曲线是一个分类器在不同的阈值下的分类效果。具体的，曲线从左往右可以认为是阈值从 0 到 1 的变化过程。当分类器阈值为 0，代表不加以识别全部判断为 0，此时 TP=FP=0，TPR=TP/P=0，FPR=FR/N=0；当分类器阈值为 1，代表不加以识别全部判断为 1，此时 FN=TN=0，P=TP+FN=TP, TPR=TP/P=1，N=FP+TN=FP, FPR=FR/N=1。所以，ROC 曲线描述的其实是分类器性能随着分类器阈值的变化而变化的过程。

对于ROC曲线，一个重要的特征是它的面积（AUC），面积为 0.5 为随机分类，识别能力为 0，面积越接近于 1 识别能力越强，面积等于 1 为完全识别。