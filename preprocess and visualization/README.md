# pandas使用
```python
# 读取文件
df = pd.read_table(
    'horse-colic.data',
    sep=' ',
    header=None,
    na_values='?',
    names=col_name,
    dtype=data_dict
)
df

# z-score归一化
def z_score_normalize(series):
    '''
    Params:  a pandas serious

    Return: normalized version of the series
    '''
    return (series-series.mean())/series.std()

# 缺失值丢弃和填充
df.dropna(thresh=threshold, inplace=True, how="all/any")
df.fillna(df.mean(), inplace=True)

# mean()平均值  median()中位数  max()最大值  min()最小值  sum()求和  std()标准差 mode()众数， 对每一列进行运算。

# 如果这些方法处理后是一个series，则直接使用：
df.fillna(df.mean())
# 如果是个单行的datafram，则使用：
df.fillna(df.mode().T[0])
# 此外，如果df是切片过的，那么不能使用inplace=True，而必须进行赋值。

# 保存文件
df.to_csv('preprocess.csv')

# 替换值
df.replace(to_replace, value)
class_mapping = {'A':0, 'B':1}
data[class] = data[class].map(class_mapping)

# 修改类型
df[[column]] = df[[column]].astype(type)

# 堆叠
pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,copy=True)
# objs：表示需要合并的表的组合[d1, d2]，接收多个Series, DataFrame, Panel 的组合，无默认；
# axis：默认为0，axis=0表示做列对齐，列名一致的话，将后表数据添加到前表的下几行；
# axis=1表示做行对齐，行标签一致的话，将后表的数据添加到前表的后几列；
# join：默认为outer，接收‘inner’或‘outer’，表示取交集或并集；

# 联表，主键合并

pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,left_index=False, 
         right_index=False,sort=False,suffixes=('_x', '_y'), copy=True,
         indicator=False,validate=None)
# left, right：分别表示需要匹配的左表和右表，可接收的数据类型为 DataFrame；
# how：表示左右表的连接方式，默认为 inner ，可接收的取值为 left、right、inner、outer；
# on：表示左右表的连接主键，两个表的主键名称一致的时候才可以使用 on 参数，不一致时需使用left_on，right_on参数， on 参数默认为None，可接收的数据类型为 str 或 sequence ；
# left_on，right_on：分别表示左表和右表的连接主键，默认为None，可接收的数据类型为 str 或 sequence ；
# sort：表示是否对合并后的数据进行排序，默认为False，可接收的数据类型为boolean ；
```
# 数据预处理
`sklearn.preprocessing.MinMaxScaler()`：归一到 [ 0，1 ] 

`sklearn.preprocessing.MaxAbsScaler()`：归一到 [ -1，1 ] 

`sklearn.preprocessing.scale()`：标准化（均值0方差1）

在Pandas中，`.duplicated()`表示找出重复的行，默认是判断全部列，返回布尔类型的结果。对于完全没有重复的行，返回 False，对于有重复的行，第一次出现的那一行返回 False，其余的返回 True。

与.duplicated()对应的，`.drop_duplicates()`表示去重，即删除布尔类型为 True的所有行，默认是判断全部列。

```py
#将工资低于1000或者高于10万的异常值清空
data[u'工资'][(data[u'工资']<1000) | (data[u'工资']>100000)] = None 
#清空后用均值插补
data.fillna(data.mean())
```

# 缺失值填充方法

## 平均值填充（Mean/Mode Completer）

将初始数据集中的属性分为数值属性和非数值属性来分别进行处理。

如果空值是数值型的，就根据该属性在其他所有对象的取值的平均值来填充该缺失的属性值；

如果空值是非数值型的，就根据统计学中的众数原理，用该属性在其他所有对象的取值次数最多的值(即出现频率最高的值)来补齐该缺失的属性值。与其相似的另一种方法叫条件平均值填充法（Conditional
Mean Completer）。在该方法中，用于求平均的值并不是从数据集的所有对象中取，而是从与该对象具有相同决策属性值的对象中取得。

这两种数据的补齐方法，其基本的出发点都是一样的，以最大概率可能的取值来补充缺失的属性值，只是在具体方法上有一点不同。与其他方法相比，它是用现存数据的多数信息来推测缺失值。

## 热卡填充（Hot deck imputation，或就近补齐）
对于一个包含空值的对象，热卡填充法在完整数据中找到一个与它最相似的对象，然后用这个相似对象的值来进行填充。不同的问题可能会选用不同的标准来对相似进行判定。该方法概念上很简单，且利用了数据间的关系来进行空值估计。这个方法的缺点在于难以定义相似标准，主观因素较多。

## K最近距离邻法（K-means clustering）

先根据欧式距离或相关分析来确定距离具有缺失数据样本最近的K个样本，将这K个值加权平均来估计该样本的缺失数据。

## 使用所有可能的值填充（Assigning All Possible values of the Attribute）

用空缺属性值的所有可能的属性取值来填充，能够得到较好的补齐效果。但是，当数据量很大或者遗漏的属性值较多时，其计算的代价很大，可能的测试方案很多。

## 组合完整化方法（Combinatorial Completer）

用空缺属性值的所有可能的属性取值来试，并从最终属性的约简结果中选择最好的一个作为填补的属性值。这是以约简为目的的数据补齐方法，能够得到好的约简结果；但是，当数据量很大或者遗漏的属性值较多时，其计算的代价很大。

## 回归（Regression）

基于完整的数据集，建立回归方程。对于包含空值的对象，将已知属性值代入方程来估计未知属性值，以此估计值来进行填充。当变量不是线性相关时会导致有偏差的估计。

## 期望值最大化方法（Expectation maximization，EM）

EM算法是一种在不完全数据情况下计算极大似然估计或者后验分布的迭代算法。在每一迭代循环过程中交替执行两个步骤：E步（Excepctaion
step,期望步），在给定完全数据和前一次迭代所得到的参数估计的情况下计算完全数据对应的对数似然函数的条件期望；M步（Maximzation
step，极大化步），用极大化对数似然函数以确定参数的值，并用于下步的迭代。算法在E步和M步之间不断迭代直至收敛，即两次迭代之间的参数变化小于一个预先给定的阈值时结束。该方法可能会陷入局部极值，收敛速度也不是很快，并且计算很复杂。

## 用随机森林拟合缺失值
```python
from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):
	#把数值型特征都放到随机森林里面去
    age_df=df[['Age','Fare','Parch','SibSp','Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y=known_age[:,0]#y是年龄，第一列数据
    x=known_age[:,1:]#x是特征属性值，后面几列
    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    #根据已有数据去拟合随机森林模型
    rfr.fit(x,y)
    #预测缺失值
    predictedAges = rfr.predict(unknown_age[:,1:])
    #填补缺失值
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    
    return df,rfr
```

## 拉格朗日插值法
```py
#拉格朗日法插补空缺值
 
import pandas as pd #导入pandas库
from scipy.interpolate import lagrange #导入拉格朗日函数
 
inputfile = u'E:\\pythondata\\cjm5.xlsx'
outputfile = u'E:\\pythondata\\cjm5_1.xlsx'
 
data= pd.read_excel(inputfile)
data[u'count'][(data[u'count']<100000) | (data[u'count']>200000)] = None #将异常值清空
 
def ployinterp_column(s,n,k=2): #k=2表示用空值的前后两个数值来拟合曲线，从而预测空值
    y = s[list(range(n-k,n)) + list(range(n+1,n+1-k))] #取值，range函数返回一个左闭右开（[left,right)）的序列数
    y = y[y.notnull()]#取上一行中取出数值列表中的非空值，保证y的每行都有数值，便于拟合函数
    return lagrange(y.index,list(y))(n) #调用拉格朗日函数，并添加索引
 
for i in data.columns: #如果i在data的列名中，data.columns生成的是data的全部列名
    for j in range(len(data)):
        if (data[i].isnull())[j]:#如果data[i][j]为空，则调用函数ployinterp_column为其插值
            data[i][j] = ployinterp_column(data[i],j)
```

# 离散化
## 等宽离散
将属性的值域从最小值到最大值分成具有相同宽度的 n 个区间，n 由数据特点决定，往往是需要有业务经验的人进行评估。
#数据离散化-等宽离散
```py
import pandas as pd
 
datafile = u'E:\\pythondata\\hk04.xlsx'
data = pd.read_excel(datafile)
data = data[u'回款金额'].copy()
k = 5 #设置离散之后的数据段为5
 
#等宽离散
d1 = pd.cut(data,k,labels = range(k))#将回款金额等宽分成k类，命名为0,1,2,3,4,5，data经过cut之后生成了第一列为索引，第二列为当前行的回款金额被划分为0-5的哪一类，属于3这一类的第二列就显示为3
```
## 等频离散
将相同数量的记录放在每个区间，保证每个区间的数量基本一致。
```py
#数据离散化-等频离散
import pandas as pd
 
datafile = u'E:\\pythondata\\hk04.xlsx'
data = pd.read_excel(datafile)
data = data[u'回款金额'].copy()
k = 5 #设置离散之后的数据段为5
 
#等频率离散化
w = [1.0*i/k for i in range(k+1)]
w = data.describe(percentiles = w)[4:4+k+1]
w[0] = w[0]*(1-1e-10)
d2 = pd.cut(data, w, labels = range(k))
```
## 聚类离散
一维聚类离散包括两个过程：通过聚类算法（K-Means 算法）将连续属性值进行聚类，处理聚类之后的到的 k 个簇，得到每个簇对应的分类值（类似这个簇的标记）。
```py
import pandas as pd
 
datafile = u'E:\\pythondata\\hk04.xlsx'
data = pd.read_excel(datafile)
data = data[u'回款金额'].copy()
k = 5 #设置离散之后的数据段为5
 
 
#聚类离散
from sklearn.cluster import KMeans
 
kmodel = KMeans(n_clusters = k, n_jobs = 4)#n_jobs是并行数，一般等于CPU数
kmodel.fit(data.reshape((len(data), 1)))
c = pd.DataFrame(kmodel.cluster_centers_, columns=list('a')).sort_values(by='a')
#rolling_mean表示移动平均，即用当前值和前2个数值取平均数，
#由于通过移动平均，会使得第一个数变为空值，因此需要使用.iloc[1:]过滤掉空值。
w = pd.rolling_mean(c, 2).iloc[1:]#此处w=[2174.1003996693553, 8547.46386803177, 22710.538501243103, 48516.861774600904]
w = [0] + list(w[0]) + [data.max()]#把首末边界点加上，首边界为0，末边界为data的最大值120000，此处w=[0, 2174.1003996693553, 8547.46386803177, 22710.538501243103, 48516.861774600904, 120000.0]
d3 = pd.cut(data, w, labels = range(k))#cut函数实现将data中的数据按照w的边界分类。
```