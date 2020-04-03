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

# 修改类型
df[[column]] = df[[column]].astype(type)



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