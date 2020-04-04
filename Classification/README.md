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

# 分类器的测度
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

## 国内通用的一些测度：
首先是AUC，表现在不同这个分类器对阈值的适应性。
```py
from sklearn import setrics
metrics.roc_auc_score(test_y,prodict_prob_y)
```
精确率、查准率：`P = TP/(TP+FP)`, “你觉得是对的当中有多少真的是对的”
召回率、查全率：`R = TP/(TP+FN)`, “对的当中有多少你看出来了”
F1 score = P和R的调和平均数，越高越好，表示模型的优秀程度。
```python
metrics.f1_score(Y_test, Y_pred, average="samples")
```
|parameter|explain|
|------|------|
|y_true | 数据真实标签 Ground truth (correct) target values.
|y_pred | 分类器分类标签 Estimated targets as returned by a classifier.
|average | [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’] 多类/多标签目标需要此参数。如果没有，则返回每个类的分数。'micro':通过先计算总体的TP，FN和FP的数量，再计算F1,'macro':分布计算每个类别的F1，然后做平均（各类别F1的权重相同）

## metric中各种评估指标
### accuracy_score
分类准确率分数是指所有分类正确的百分比。分类准确率这一衡量分类器的标准比较容易理解，但是它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。

形式：
sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)

normalize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数

————————————————

### recall_score
召回率 =提取出的正确信息条数 /样本中的信息条数。通俗地说，就是所有准确的条目有多少被检索出来了。

形式：
klearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1,average='binary', sample_weight=None)

参数average : string, [None, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’]

将一个二分类matrics拓展到多分类或多标签问题时，我们可以将数据看成多个二分类问题的集合，每个类都是一个二分类。接着，我们可以通过跨多个分类计算每个二分类metrics得分的均值，这在一些情况下很有用。你可以使用average参数来指定。

macro：计算二分类metrics的均值，为每个类给出相同权重的分值。当小类很重要时会出问题，因为该macro-averging方法是对性能的平均。另一方面，该方法假设所有分类都是一样重要的，因此macro-averaging方法会对小类的性能影响很大。

weighted:对于不均衡数量的类来说，计算二分类metrics的平均，通过在每个类的score上进行加权实现。

micro：给出了每个样本类以及它对整个metrics的贡献的pair（sample-weight），而非对整个类的metrics求和，它会每个类的metrics上的权重及因子进行求和，来计算整个份额。Micro-averaging方法在多标签（multilabel）问题中设置，包含多分类，此时，大类将被忽略。

samples：应用在multilabel问题上。它不会计算每个类，相反，它会在评估数据中，通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）

average：average=None将返回一个数组，它包含了每个类的得分.
————————————————

### roc_curve
ROC曲线指受试者工作特征曲线/接收器操作特性(receiver operating characteristic，ROC)曲线,是反映灵敏性和特效性连续变量的综合指标,是用构图法揭示敏感性和特异性的相互关系，它通过将连续变量设定出多个不同的临界值，从而计算出一系列敏感性和特异性。ROC曲线是根据一系列不同的二分类方式（分界值或决定阈），以真正例率（也就是灵敏度）（True Positive Rate,TPR）为纵坐标，假正例率（1-特效性）（False Positive Rate,FPR）为横坐标绘制的曲线。

ROC观察模型正确地识别正例的比例与模型错误地把负例数据识别成正例的比例之间的权衡。TPR的增加以FPR的增加为代价。ROC曲线下的面积是模型准确率的度量，AUC（Area under roccurve）。

纵坐标：真正率（True Positive Rate , TPR）或灵敏度（sensitivity）

TPR = TP /（TP + FN）  （正样本预测结果数 / 正样本实际数）

横坐标：假正率（False Positive Rate , FPR）

FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）

形式：
sklearn.metrics.roc_curve(y_true,y_score, pos_label=None, sample_weight=None, drop_intermediate=True)

该函数返回这三个变量：fpr,tpr,和阈值thresholds;

这里理解thresholds:

分类器的一个重要功能“概率输出”，即表示分类器认为某个样本具有多大的概率属于正样本（或负样本）。

“Score”表示每个测试样本属于正样本的概率。

接下来，我们从高到低，依次将“Score”值作为阈值threshold，当测试样本属于正样本的概率大于或等于这个threshold时，我们认为它为正样本，否则为负样本。每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点。当我们将threshold设置为1和0时，分别可以得到ROC曲线上的(0,0)和(1,1)两个点。将这些(FPR,TPR)对连接起来，就得到了ROC曲线。当threshold取值越多，ROC曲线越平滑。其实，我们并不一定要得到每个测试样本是正样本的概率值，只要得到这个分类器对该测试样本的“评分值”即可（评分值并不一定在(0,1)区间）。评分越高，表示分类器越肯定地认为这个测试样本是正样本，而且同时使用各个评分值作为threshold。我认为将评分值转化为概率更易于理解一些。
————————————————

### Auc
计算AUC值，其中x,y分别为数组形式，根据(xi,yi)在坐标上的点，生成的曲线，然后计算AUC值；

形式：

sklearn.metrics.auc(x, y, reorder=False)

————————————————

### roc_auc_score
直接根据真实值（必须是二值）、预测值（可以是0/1,也可以是proba值）计算出auc值，中间过程的roc计算省略。

形式：
sklearn.metrics.roc_auc_score(y_true, y_score, average='macro', sample_weight=None)

average : string, [None, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’]
————————————————


### confusion_matrix

形式：
sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

返回一个混淆矩阵；

labels：混淆矩阵的索引（如上面猫狗兔的示例），如果没有赋值，则按照y_true, y_pred中出现过的值排序。
————————————————
sklearn中的classification_report函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。 
主要参数: 
y_true：1维数组，或标签指示器数组/稀疏矩阵，目标值。 
y_pred：1维数组，或标签指示器数组/稀疏矩阵，分类器返回的估计值。 
labels：array，shape = [n_labels]，报表中包含的标签索引的可选列表。 
target_names：字符串列表，与标签匹配的可选显示名称（相同顺序）。 
sample_weight：类似于shape = [n_samples]的数组，可选项，样本权重。 
digits：int，输出浮点值的位数．

# 多标签和多类别的处理

## One VS Rest 策略
这个策略同时也称为One-vs-all策略，即通过构造K个判别式（K为类别的个数），第i个判别式将样本归为第i个类别或非第i个类别。
### 多类别
```py
from sklearn.multiclass import OneVsRestClassifier

iris = datasets.load_iris()
X,y = iris.data,iris.target
OneVsRestClassifier(LinearSVC(random_state = 0)).fit(X,y).predict(X)
```
### 多标签
如果是多标签，也就是一个样本可能同时属于多个分类，多一步把`[[1],[2,3]]`类型的label变成`[[1,0,0],[0,1,1]]`：
```python
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
# transform y into a matrix
mb = MultiLabelBinarizer()
y_train = mb.fit_transform(y_train)

# fit the model and predict
clf = OneVsRestClassifier(LogisticRegression(),n_jobs=-1)
clf.fit(X_train,y_train)
pred_y = clf.predict(X_test)
```
## One VS One 策略
One-Vs-One策略即是两两类别之间建立一个判别式，这样，总共需要K(K−1)/2个判别式，最后通过投票的方式确定样本所属类别。

只要把上面的`OneVsRestClassifier`替换为`OneVsOneClassifier`即可。