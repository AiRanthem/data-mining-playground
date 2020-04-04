<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# TODO
1. 修改公式显示脱离脚本
2. 重写使用工具库函数的代码，改为使用sklearn
# 概念
$$Support(X,Y) = P(XY) = \frac{number(XY)}{num(All Samples)}$$
$$Confidence(X \Leftarrow Y) = P(X|Y)=P(XY)/P(Y)$$
$$Lift(X \Leftarrow Y) = P(X|Y)/P(X) = Confidence(X \Leftarrow Y) / P(X)$$
# 工具库
## [Apriori.py](./libs/Apriori.py)
### apriori(dataSet,minSupport=0.5)
+ params:
    + **dataset**: 2d list
    + **minSupport**: support threshold
+ return:
    + **L**: list. L[k-1] = \\(L_k\\)
    + **supportData**: a dict maps all itemset->sup (for all including those not frequent)
### generateRules(L,suppData, minConf=0.7)
+ params:
    + **L**: list. L[k-1] = \\(L_k\\)
    + **supportData**: a dict maps all itemset->sup (for all including those not frequent)
    + **minConf**: confidence threshold
+ return:
    + **rules**: list of tuples in present of sets quailified
## [fpgroth.py](./libs/fpgroth.py)
### find_frequent_patterns(transactions, support_threshold)
+ params:
    + **transactions**: 2d list, each row is a row in df
    + **support_threshold**: *here it is the occurances time not the support.*
+ return:
    + **patterns**: key:a pattern, value: occurrences time
### generate_association_rules(patterns, confidence_threshold)
+ params:
    + **patterns**: key:a pattern, value: occurrences time
    + **confidence_threshold**
+ return:
    + **rules**: key: itemset; value: support number
    ```Ipython
    # in
    for i in rules:
        print(i, ' => ', rules[i])
    # out
    ('eggs', 'ground beef', 'pancakes', 'spaghetti')  =>  (('nan',), 9.875)
    ('chocolate', 'eggs', 'ground beef', 'pancakes', 'spaghetti')  =>  (('nan',), 10.2)
...
    ```
# Aprior算法
![](https://zty-pic-bed.oss-cn-shenzhen.aliyuncs.com/20200320125536.png)
+ 输入：数据集合D，支持度阈值α
+ 输出：最大的频繁k项集

1. 扫描整个数据集，得到所有出现过的数据，作为候选频繁1项集。k=1，频繁0项集为空集。
2. 挖掘频繁k项集
    1. 扫描数据计算候选频繁k项集的支持度
    2. 去除候选频繁k项集中支持度低于阈值的数据集,得到频繁k项集。如果得到的频繁k项集为空，则直接返回频繁k-1项集的集合作为算法结果，算法结束。如果得到的频繁k项集只有一项，则直接返回频繁k项集的集合作为算法结果，算法结束。
    3. 基于频繁k项集，连接生成候选频繁k+1项集。
3. 令k=k+1，转入步骤2。

从算法的步骤可以看出，Aprior算法每轮迭代都要扫描数据集，因此在数据集很大，数据种类很多的时候，算法效率很低。

# FP growth算法
## 构建FP树
步骤1:
1. 遍历所有的数据集合，计算所有项的支持度。
2. 丢弃非频繁的项。
3. 基于 支持度 降序排序所有的项。
4. 所有数据集合按照得到的顺序重新整理。
5. 重新整理完成后，丢弃每个集合末尾非频繁的项。
![](https://zty-pic-bed.oss-cn-shenzhen.aliyuncs.com/20200320144748.png)
![](https://zty-pic-bed.oss-cn-shenzhen.aliyuncs.com/20200320144908.png)

步骤2: 读取每个集合插入FP树中，同时用一个头部链表数据结构维护不同集合的相同项。
1. 构造过程
![构造过程](https://zty-pic-bed.oss-cn-shenzhen.aliyuncs.com/20200320145006.png)
2. 结果
![最终效果](https://zty-pic-bed.oss-cn-shenzhen.aliyuncs.com/20200320145029.png)

## 挖掘模式
从根节点开始遍历，每一条路径就是一个频繁项集，其叶节点的值就是频数。
