---
layout: post
title: Classification
date: 2021-09-19 19:20
categories: [Hands-On Machine Learning]
tags: [Classification,MachineLearning]
---
# MNIST数据集

MINST数据数据集是手工写的70000个数字图片。每张图片都表示其代表的数字标记。

Scikit-Learn提供了下载此数据集


```python
# 获取MNIST数据集
from sklearn.datasets import fetch_openml
'''
Warning: since Scikit-Learn 0.24, fetch_openml() returns a Pandas DataFrame by default. 
To avoid this and keep the same code as in the book, we use as_frame=False.
'''
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()
```




    dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])



Scikit-Learn加载的数据集通常具有类似的字典结构
- DESCR键：描述数据集。
- data键：包含一个数组，map个实例为一行，每个特征为一列。
- target键：包含一个带有标记的数组。


```python
# 查看数组
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)
```

    (70000, 784)
    (70000,)
    

共有7万张图片，每张图片有784个特征（图片是28*28=784像素），每个特征代表着一个像素点的强度，从 0-255 。


```python
# 查看任意一张图片
import matplotlib as mpl
import matplotlib.pyplot as plt

# 选取一个实例的特征向量，将其转换成28*28数组
some_digit = X[0]
print(some_digit)
some_digit_image = some_digit.reshape(28, 28)

# 使用Matplotlib的imshow()函数进行显示
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()
```

    [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   3.  18.
      18.  18. 126. 136. 175.  26. 166. 255. 247. 127.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.  30.  36.  94. 154. 170. 253.
     253. 253. 253. 253. 225. 172. 253. 242. 195.  64.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.  49. 238. 253. 253. 253. 253. 253.
     253. 253. 253. 251.  93.  82.  82.  56.  39.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.  18. 219. 253. 253. 253. 253. 253.
     198. 182. 247. 241.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.  80. 156. 107. 253. 253. 205.
      11.   0.  43. 154.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.  14.   1. 154. 253.  90.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 139. 253. 190.
       2.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  11. 190. 253.
      70.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  35. 241.
     225. 160. 108.   1.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  81.
     240. 253. 253. 119.  25.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
      45. 186. 253. 253. 150.  27.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.  16.  93. 252. 253. 187.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0. 249. 253. 249.  64.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
      46. 130. 183. 253. 253. 207.   2.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  39. 148.
     229. 253. 253. 253. 250. 182.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  24. 114. 221. 253.
     253. 253. 253. 201.  78.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.  23.  66. 213. 253. 253. 253.
     253. 198.  81.   2.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.  18. 171. 219. 253. 253. 253. 253. 195.
      80.   9.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.  55. 172. 226. 253. 253. 253. 253. 244. 133.  11.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0. 136. 253. 253. 253. 212. 135. 132.  16.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
       0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
    


    
![png](/assets/images/img_HandsOnML/output_6_1.png)
    



```python
# 标签
y[0]
```




    '5'




```python
# 将y转换成整数
import numpy as np

y = y.astype(np.uint8)
```


```python
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")
```


```python
# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
```


```python
plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
plt.show()
```


    
![png](/assets/images/img_HandsOnML/output_11_0.png)
    



```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

# 训练二元分类器

现在先简化一下问题，只尝试识别一个数字。比如数字5，那么数字五检测器就是一个二元分类检测器，他只能识别5和非5.

首先，为此分类任务创建目标向量


```python
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
```


```python
# 使用随机梯度下降（SGD）分类器进行训练
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```




    SGDClassifier(random_state=42)



> 说明 ：SGDClassifier 在训练的时候是完全随机的，为了使得结果可以复现，需要设置参数 random_state

> 随机梯度下降：每次迭代都随机从训练集中抽取出1个样本，在样本量极其大的情况下，可能不用抽取出所有样本，就可以获得一个损失值在可接受范围之内的模型了。缺点是由于单个样本可能会带来噪声，导致并不是每次迭代都向着整体最优方向前进。


```python
# 检测数字5
sgd_clf.predict([some_digit])
```




    array([ True])



# 性能测量

## 使用交叉验证测量准确率


```python
# 实现交叉验证。类似于Scikit-Learn提供的cross_val_score()这一类交叉验证函数
'''
k折交叉验证 K-fold Cross Validation(记为K-CV)

1. 将数据集平均分割成K个等份（参数cv值，一般选择5折10折，即测试集为20%）
2. 使用1份数据作为测试数据，其余作为训练数据
3. 计算测试准确率
4. 使用不同的测试集，重复2、3步
5. 对测试准确率做平均，作为对未知数据预测准确率的估计

优点： 
    因为每一个样本数据既可以作为测试集又可以作为训练集，可有效避免欠学习和过学习状态的发生，得到的结果比较有说服力。
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
'''
StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
'''
skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
```

    0.9669
    0.91625
    0.96785
    

使用cross_val_score()函数来评估SGDClassifier模型。


```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy")
```




    array([0.95035, 0.96035, 0.9604 ])




```python
#如果把每张图都分类成“非5”：
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```




    array([0.91125, 0.90855, 0.90915])



> 说明：在这种情况下，准确率依然高达90% ，这是因为只有大约10%的图片是5，所以如果猜一张图片是 非5 的准确率高达90% 。这样显然，**准确率通常无法作为分类器的首要性能指标**。

## 混淆矩阵

评估分类器性能更好的方法是混淆矩阵。其总体思路是统计A类别实例被识别为B类别的次数。例如，要想知道分类器将数字3和数字5混淆了多少次，只需要通过混淆矩阵的第五行第三列来进行查看。


```python
# 在计算混淆矩阵之前，需要有一组预测才能将其与实际目标进行比较
from sklearn.model_selection import cross_val_predict

y_train_pred=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
```

> 说明：与cross_val_score函数一样，cross_val_predict()函数也是同时执行K-折交叉验证，但是返回的不是评估分数而是每一个折叠的预测。这样可以得到一个干净（干净：指的是模型预测时使用的数据在其在其训练期间从未使用过）的预测。


```python
# 使用confusion_matrix() 函数来获取混淆矩阵
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5,y_train_pred)
```




    array([[53892,   687],
           [ 1891,  3530]], dtype=int64)



> 说明：混淆矩阵中行表示实际类别，列表示预测类别。第一行表示：在所有的 “非5（负类）”的图片中:53892张被正确的识别为 ”非5“类别（真负类），687张被错误的分类成了 “5” （假负类）。;第二行表示：所有的 ”5“（正类）的图片中1891张被错误的识别为 ”非5“(假负类），3530被正确的识别为 "5"这一类别（真正类）。

一个完美的分类器只有真正类**（被正确的识别为5）**与真负类**（被正确的识别为非5）**。


```python
y_train_prefect_predictions=y_train_5
confusion_matrix(y_train_5,y_train_prefect_predictions)
```




    array([[54579,     0],
           [    0,  5421]], dtype=int64)



**公式3-1：精度（*precision*)**
$$precision={TP \over TP+FP}$$

   TP:真正类的数量。FP：假正类的数量。

**公式3-2：召回率（*recall*）**
$$recall={TP\over TP+FN}$$
    FN:假负类数量
 
混淆矩阵图片说明。真负（左上），假正（右上），假负（左下），真正（右下）
    ![混淆矩阵](/assets/images/img_HandsOnML/图3-2.jpg)

> **精度与召回率总结**：精度表示:正确识别为5的数量/(正确识别为5的数量+本不应该识别识别为5却识别为5的数量），这也就意味着精度越高识别为5时，这个图片确实为5的概率较高。
召回率表示：所有5中正确识别的比率。意味着召回率越高，5的识别率较高。

## 精度和召回率


```python
# 精度与召回率
from sklearn.metrics import precision_score,recall_score

print(precision_score(y_train_5,y_train_pred))

print(recall_score(y_train_5,y_train_pred))
```

    0.8370879772350012
    0.6511713705958311
    

将精度与召回率组成一个单一的指标称为F1指标。F1分数是精度与召回率的谐波平均值。谐波平均值会给予低值更好的权重，只有当精度与召回率都很高时才会很高。

**公式3-3：F1**

$$F_1= {2\over { {1\over 精度}+{1\over 召回率} } }=2*{ {精度*召回率}\over{精度+召回率} }={TP\over {TP+{ {FN+FP}\over2} } } $$
    


```python
# 计算F1分数
from sklearn.metrics import f1_score

f1_score(y_train_5,y_train_pred)
```




    0.7325171197343846



## 精度与召回率权衡

在精度与召回率的权衡中，图像按其分类器评分进行排名，而高于所选决策阈值的图像被认为是正的；阈值越高召回率越低，但是通常精度较高。


   ![图3-3](/assets/images/img_HandsOnML/图3-3.jpg)

Scikit-Learn 不允许直接设置阈值，但是可以访问他用于预测的决策分数。通过调用decision_function()方法实现，这种方法返回每个实例的分数，然后可以根据分数使用任意阈值进行预测。


```python
# 获取预测的决策分数
y_scores=sgd_clf.decision_function([some_digit])
y_scores
```




    array([2164.22030239])




```python
# 设置阈值为零
threshold=0
y_some_digit_pred=(y_scores>threshold)
```

SGDClassifier分类器设置的阈值是零。代码返回结果与predict()预测结果是一样的所有返回True


```python
# 设置阈值我8000
threshold=8000
y_some_digit_pred=(y_scores>threshold)
y_some_digit_pred
```




    array([False])



提高阈值可以降低召回率（所有的5，但是检测出来较少）。这张图确实是5，但是阈值为0时可以检测到，但是当阈值为8000时，就错过了这张图。

**决定使用什么阈值**


```python
# 首先，使用cross_val_predict()函数来获取所有实例的分数，返回的是决策分数而不是预测结果
y_scores=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")
```


```python
# 有了分数之后，可以使用precison_recall_curve()函数来计算所有可能的阈值的精度与召回率
from sklearn.metrics import precision_recall_curve

precisions,recalls,thresholds=precision_recall_curve(y_train_5,y_scores)
```


```python
# 使用Matplotlib绘制精度召回率相对于阈值的函数图
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) 
    plt.xlabel("Threshold", fontsize=16)        
    plt.grid(True)                              
    plt.axis([-50000, 50000, 0, 1])             

recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

plt.figure(figsize=(8, 4))                                                                  # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 # Not shown
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                # Not shown
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
plt.plot([threshold_90_precision], [0.9], "ro")                                             # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown 
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()
```


    
![png](/assets/images/img_HandsOnML/output_52_0.png)
    



```python
# 提供至少90%精度的最低阈值
threshold_90_precision=thresholds[np.argmax(precisions>=0.9)]
print(threshold_90_precision)
```

    3370.0194991439557
    


```python
# 在训练集进行预测
y_train_pred_90=(y_scores>=threshold_90_precision)
```


```python
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
plot_precision_vs_recall(precisions,recalls)
plt.show()
```


    
![png](/assets/images/img_HandsOnML/output_55_0.png)
    



```python
# 检测一下预测结果的精度与召回率
precision_score(y_train_5,y_train_pred_90)
```




    0.9000345901072293




```python
recall_score(y_train_5,y_train_pred_90)
```




    0.4799852425751706



## ROC曲线

还有一种与二元分类器一起使用的工具称之为受试者工作特征曲线（简称ROC）
ROC曲线绘制的是正真类率（召回率）与假正类率（FPR）。**FPR：** 是被错误分为正类的负类实例比率。他等于1减去真负类率（TNR），后者是被正确分类为负类实例比率特称之为特异度。

ROC曲线是绘制灵敏度（召回率）与（1-特异度）的关系。

**绘制ROC曲线**


```python
# 首先使用roc_curve()函数来计算多种阈值的TPR和FPR
from sklearn.metrics import roc_curve

fpr,tpr,thresholds=roc_curve(y_train_5,y_scores)
```


```python
# 绘制ROC曲线
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

plt.figure(figsize=(8, 6))                                    
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]          
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")               
plot_roc_curve(fpr,tpr)                                 
plt.show()
```


    
![png](/assets/images/img_HandsOnML/output_62_0.png)
    


> 说明：该曲线绘制了所有可能阈值的假正率和真正率的关系。粗点处突出显示了选定的比率（召回率为43.68%）

测量曲线下面积（AUC）。完美分类器的ROC AUC等与 1，而纯随机的分类器的ROC AUC等与0.5.


```python
# 计算AUC
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5,y_scores)
```




    0.9604938554008616



> ROC曲线与精度/召回率(PR)曲线的选择:当正类非常少见或者你更关注假正类而不是假负类时选择PR曲线，反之选择ROC曲线

**训练RandomForestClassifier分类器**

RandomForestClassifier类中没有decision_function()方法。相反，它有predict_proba()方法。

predict_proba()方法会返回一个数组，其中每行代表一个实例，每列表示一个类别，意思是某个给定实例属于某个 给定类别的概率（例如，这张图片有70%可能是数字5）


```python
from sklearn.ensemble import RandomForestClassifier

forest_clf =RandomForestClassifier(random_state=42)
y_probas_forest =cross_val_predict(forest_clf,X_train,y_train_5,cv=3,
                                   method="predict_proba")
```


```python
# roc_curve()函数需要标签和分数，但是我们不提供分数而是提供概率。在这里直接使用概率作为分数
y_scores_forest =y_probas_forest[:,1]
fpr_forest,tpr_forest,thresholds_forest=roc_curve(y_train_5,y_scores_forest)
```


```python
# 绘制图像
plt.plot(fpr,tpr,"b:",label="SGD")
plot_roc_curve(fpr_forest,tpr_forest,"Random Forest")
plt.legend(loc="lower right")
plt.show()
```


    
![png](/assets/images/img_HandsOnML/output_71_0.png)
    


> 比较ROC曲线：随机森林分类器要优于SGD分类器，因为它的ROC曲线更靠近左上角并且具有更大的AUC。


```python
# ROC AUC分数
roc_auc_score(y_train_5,y_scores_forest)
```




    0.9983436731328145



# 多类分类器

创建一个系统可以将数字图片分为10类：

**方法一: 一对剩余（OVR）策略**

训练十个分类器，用来检测0~10。当需要检测图片时，获取每个分类器的决策分数，输出最高分数。

**方法二:一对一（OVO）策略**

为每一个数字都训练一个二元分类器：一个用于区分0和1，一个区分0和2，等。最终需要训练 N*(N-1)/2个分类器。

Scikit-Learn可以检测到你尝试使用二元分类算法进行多分类任务时，它会自动选择）OVR或OVO。


```python
# 使用sklean.svm.svc类来创建SVM分类器
from sklearn.svm import SVC

svm_clf=SVC() #创建分类器对象
svm_clf.fit(X_train,y_train) #用训练数据拟合分类器模型
svm_clf.predict([some_digit]) # 用训练好的分类器去预测[some_digit]数据的标签[5]
```




    array([5], dtype=uint8)



> 这段代码使用原始目标类0~9（y_train)在训练集上进行SVM训练。在其内部，Scikit-Learn实际上训练了45个分类器，获取他们对图片的局的分数，然后选择最高分数。


```python
# 调用decision_function()
some_digit_scores=svm_clf.decision_function([some_digit])
some_digit_scores
```




    array([[ 1.72501977,  2.72809088,  7.2510018 ,  8.3076379 , -0.31087254,
             9.3132482 ,  1.70975103,  2.76765202,  6.23049537,  4.84771048]])




```python
# 返回的是最大数的索引 some_digit_scores[5]= 9.3132482(值最大，返回索引)
np.argmax(some_digit_scores)
```




    5




```python
svm_clf.classes_
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)




```python
svm_clf.classes_[5]
```




    5



> 当训练分类器时，目标的列表会存储到classes_ 属性中，安值得大小排序


```python
# 强制Scikit-Learn使用一对一或一对剩余。可以使用 OneVsOne-Classifier或OneVsRestClassifier类。
from sklearn.multiclass import OneVsRestClassifier

ovr_clf=OneVsRestClassifier(SVC())
ovr_clf.fit(X_train[:1000],y_train[:1000])
ovr_clf.predict([some_digit])
```




    array([5], dtype=uint8)




```python
len(ovr_clf.estimators_)
```




    10




```python
# 训练SGDClassifier 或者RandomForestClassifier
sgd_clf.fit(X_train,y_train)
sgd_clf.predict([some_digit])
```




    array([3], dtype=uint8)




```python
# SGD 分类器直接就可以将实例分为多个类，调用 decision_function()获得分类器将每个实例分类为每个类的概率列表：
sgd_clf.decision_function([some_digit])
```




    array([[-31893.03095419, -34419.69069632,  -9530.63950739,
              1823.73154031, -22320.14822878,  -1385.80478895,
            -26188.91070951, -16147.51323997,  -4604.35491274,
            -12050.767298  ]])




```python
# 交叉验证来评估SGDClassifier的准确性
cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring="accuracy")
```




    array([0.87365, 0.85835, 0.8689 ])




```python
# 将输入进行简单缩放，来提高准确率
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring="accuracy")
```




    array([0.8983, 0.891 , 0.9018])



# 误差分析


```python
# 混淆矩阵 cross_val_predict() 函数进行预测，然后调用 confusion_matrix()函数:
y_train_pred=cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
conf_mx=confusion_matrix(y_train,y_train_pred)
conf_mx
```




    array([[5577,    0,   22,    5,    8,   43,   36,    6,  225,    1],
           [   0, 6400,   37,   24,    4,   44,    4,    7,  212,   10],
           [  27,   27, 5220,   92,   73,   27,   67,   36,  378,   11],
           [  22,   17,  117, 5227,    2,  203,   27,   40,  403,   73],
           [  12,   14,   41,    9, 5182,   12,   34,   27,  347,  164],
           [  27,   15,   30,  168,   53, 4444,   75,   14,  535,   60],
           [  30,   15,   42,    3,   44,   97, 5552,    3,  131,    1],
           [  21,   10,   51,   30,   49,   12,    3, 5684,  195,  210],
           [  17,   63,   48,   86,    3,  126,   25,   10, 5429,   44],
           [  25,   18,   30,   64,  118,   36,    1,  179,  371, 5107]],
          dtype=int64)




```python
# 图像显示
plt.matshow(conf_mx,cmap=plt.cm.gray)
plt.show()
```


    
![png](/assets/images/img_HandsOnML/output_92_0.png)
    


> 说明：通过混淆矩阵图片发现，大多数图片都在主对角线上，这说明他们被正确的分类。

现在，我们考虑一下错误率（错误数量的绝对值不太公平），将混淆矩阵中的每个值都除以相应类的图片数量，这就比较了错误率而不是错误的绝对值


```python
row_sums=conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx=conf_mx/row_sums
# 用0填充对角线，仅保存错误重新绘制
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
plt.show()
```


    
![png](/assets/images/img_HandsOnML/output_95_0.png)
    


> 说明：通过图像可以很清晰的看到错误类型。记住，每一行代表实际类，每一列代表预测类。第八列非常的亮，代表许多图片都被错误的识别成了8。

现在，可以将更多的精力花在改进数字8的分类器。可以通过以下方法改进

- 收集更多看起来像数字8的训练数据，以便分类器可以更好的将他们与真实数字区分开来。
- 可以下一个算法来计算闭环的数量
- 对图片进行预处理

查看一下数字3和5的实例（plot_digits()函数只是使用了Matplotlib的imshow()函数


```python
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()
```


    
![png](/assets/images/img_HandsOnML/output_98_0.png)
    


> 左侧两个5*5的矩阵显示了被分类成了数字3的图片，右侧的两个矩阵显示了被分类成了数字5的图片。原因在于：我们使用的是简单的SGDClassifier模型是线性模型。她所作的工作就是为每一个像素分配一个各个类别的的权重，当他看到新的图像时，将加权后的像素强度汇总，从而得到一个分数进行分类。

> 数字3和数字5 主要的区别在于连接顶线和下方弧线的中间一小段线条的位置。如果写三时连接点稍微左移分类器就会将其是别为数字5，反之亦然。换言之，这个分类器对图像的旋转和位移非常敏感。处理方法：对图像进行预处理，确保他们位于中心位置并且没有旋转。

# 多标签分类

到目前为止，每个实例都只会被分在一个类里，在某些情况下你希望分类器为每个实例输出多个类。


```python
# 简单的实例
from sklearn.neighbors import KNeighborsClassifier

y_train_large=(y_train>=7)
y_train_odd=(y_train%2==1)
y_multilabel=np.c_[y_train_large,y_train_odd]

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)
```




    KNeighborsClassifier()



> 说明： 这段代码会创建一个 y_multilabel数组，其中包含两个数字图片的目标标签：第一个表示数字是否为大数（7，8，9），第二个标签表示是否为奇数。下一行，创建KNeighborsClassifier实例（他支持多标签分类，不是所有的实例都支持多标签分类），然后使用多个目标数组对他进行训练。


```python
# 预测
knn_clf.predict([some_digit])
```




    array([[False,  True]])



> 数字 5 ，确实不大且为奇数


```python
# 评估多标签分类器

# 计算所有标签的平均F1分数
y_train_knn_pred=cross_val_predict(knn_clf,X_train,y_multilabel,cv=3)
f1_score(y_multilabel,y_train_knn_pred,average="macro")
```




    0.976410265560605



# 多输出分类（多输出-多类分类）

多输出分类：他是多标签分类的泛化，其标签也可以是多类的。

为了说明这一点，构造一个系统去除图片中噪声。


```python
# 首先从创建测试集与训练集开始，使用Numpy的randint()函数为MNIST图片的像素强度增加噪点。
noise=np.random.randint(0,100,(len(X_train),784))
X_train_mod=X_train+noise
noise=np.random.randint(1,100,(len(X_test),784)) 
X_test_mod=X_test+noise
y_train_mod=X_train
y_test_mod=X_test
```


```python
# 通过训练分类器，清洗这张图片
some_index=0
knn_clf.fit(X_train_mod,y_train_mod)
clean_digit=knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
```


    
![png](/assets/images/img_HandsOnML/output_110_0.png)
    


# 练习

## Tackle the Titanic dataset

目标是根据乘客的年龄,性别,乘客等级,登船地点等属性来预测乘客是否存活

首先,需要先获取[数据](https://github.com/laputa-coder/AI-Guide/tree/main/MachineLearning/Hands-On%20Machine%20Learning/datasets/titanic),下载 train.csv和test.csv .并保存到 datasets/titanic 文件夹.


```python
"""
加载数据
"""
import os

TITANIC_PATH=os.path.join("datasets/","titanic")

```


```python
import pandas as pd 

def load_titanic_data(filename,titanic_path=TITANIC_PATH):
    csv_path=os.path.join(titanic_path,filename)
    return pd.read_csv(csv_path)
```


```python
train_data=load_titanic_data("train.csv")
test_data=load_titanic_data("test.csv")
```

数据已经分成训练集和测试集。 然而，测试数据并不包含标签:您的目标是使用训练数据训练出最佳模型，然后对测试数据进行预测，并将它们上传到Kaggle以查看您的最终分数。


```python
"""
首先看一下训练集前几行
"""
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



首先,看一下列表属性:
- Survived:这是目标,0表示乘客没有幸存,而1表示幸存下来.
- Pclass:乘客舱
- Name,Sex,Age :不言自明
- SibSp:乘客的兄弟姐妹和配偶数量
- Parch:乘客的孩子和父母数量
- Ticket:ticket id
- Fare:票价
- Cabin:客舱号
- Embarked:登船地点


```python
"""
获取更多信息来查看丢失了多少数据
"""
train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    

Age、Cabin 和 Embarked 属性有时为空（小于 891 非空），尤其是 Cabin（77% 为空）。 我们现在将忽略 Cabin 并专注于其余部分。 Age 属性有大约 19% 的空值，因此我们需要决定如何处理它们。 用年龄中位数替换空值似乎是合理的。

Name 和 Ticket 属性可能有一些值，但将它们转换为模型可以使用的有用数字会有点棘手。 所以现在，我们将忽略它们。



```python
"""
我们来看看数值属性：
"""
train_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



只有38%的人幸存了下来。 (这已经非常接近40%了，所以准确度将是评估我们模型的一个合理指标。  
平均票价是32.20英镑，这看起来并不贵(但在当时可能是一大笔钱)。  
平均年龄在30岁以下。


```python
"""
让我们检查一下目标确实是 0 或 1
"""
train_data["Survived"].value_counts()
```




    0    549
    1    342
    Name: Survived, dtype: int64




```python
train_data["Pclass"].value_counts()
```




    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64




```python
train_data["Sex"].value_counts()
```




    male      577
    female    314
    Name: Sex, dtype: int64




```python
train_data["Embarked"].value_counts()
```




    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64



Embarked 属性告诉我们乘客从哪里上船:C=Cherbourg,Q=Queenstown,S=Southampton
    
注意:下面的代码混合使用了Pipeline、FeatureUnion和自定义DataFrameSelector来以不同的方式预处理一些列。 由于Scikit-Learn 0.20，最好使用ColumnTransformer，就像在前一章中那样。  
 
现在让我们构建预处理管道。 我们将重用在上一章中构建的DataframeSelector来从DataFrame中选择特定的属性:  


```python
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
```


```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])
```


```python
num_pipeline.fit_transform(train_data)
```




    array([[22.    ,  1.    ,  0.    ,  7.25  ],
           [38.    ,  1.    ,  0.    , 71.2833],
           [26.    ,  0.    ,  0.    ,  7.925 ],
           ...,
           [28.    ,  1.    ,  2.    , 23.45  ],
           [26.    ,  0.    ,  0.    , 30.    ],
           [32.    ,  0.    ,  0.    ,  7.75  ]])




```python
"""
我们还需要一个用于字符串分类列的输入器（常规 SimpleImputer 不适用于那些）：
"""
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)
```


```python
from sklearn.preprocessing import OneHotEncoder
```


```python
"""
现在我们可以为分类属性构建管道
"""
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
```


```python
cat_pipeline.fit_transform(train_data)
```




    array([[0., 0., 1., ..., 0., 0., 1.],
           [1., 0., 0., ..., 1., 0., 0.],
           [0., 0., 1., ..., 0., 0., 1.],
           ...,
           [0., 0., 1., ..., 0., 0., 1.],
           [1., 0., 0., ..., 1., 0., 0.],
           [0., 0., 1., ..., 0., 1., 0.]])




```python
"""
最后，让我们加入数值和分类管道：
"""
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
```


```python
"""
酷！现在我们有一个很好的预处理管道，它获取原始数据并输出我们可以提供给我们想要的任何机器学习模型的数字输入特征。
"""
X_train = preprocess_pipeline.fit_transform(train_data)
X_train
```




    array([[22.,  1.,  0., ...,  0.,  0.,  1.],
           [38.,  1.,  0., ...,  1.,  0.,  0.],
           [26.,  0.,  0., ...,  0.,  0.,  1.],
           ...,
           [28.,  1.,  2., ...,  0.,  0.,  1.],
           [26.,  0.,  0., ...,  1.,  0.,  0.],
           [32.,  0.,  0., ...,  0.,  1.,  0.]])




```python
"""
获取标签
"""
y_train = train_data["Survived"]
```


```python
"""
我们现在准备训练分类器。让我们从 SVC 开始
"""
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)
```




    SVC(gamma='auto')




```python
"""
我们的模型已经训练好了，让我们用它来对测试集进行预测：
"""
X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)
```


```python
"""
现在我们可以根据这些预测构建一个CSV文件(考虑到Kaggle除外的格式)，然后上传它，并期待最好的结果。
但是等等! 我们能做得比希望更好。 为什么我们不使用交叉验证来了解我们的模型有多好?  
"""
from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()
```




    0.7329588014981274



好吧，超过73%的准确率，明显比随机评分好，但这不是一个好分数。 看看Kaggle上《泰坦尼克号》的排行榜，你可以看到你需要达到80%以上的准确率才能进入前10%的Kagglers。 有些达到了100%，但由于你可以很容易地找到泰坦尼克号的受害者名单，似乎机器学习在他们的表现中几乎没有涉及! 所以让我们试着建立一个精确度达到80%的模型。  


```python
"""
Let's try a RandomForestClassifier:
"""
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
```




    0.8126466916354558



为了进一步改善这个结果，您可以:  
 
比较更多的模型，使用交叉验证和网格搜索优化超参数，  
做更多的功能工程，例如:  
用它们的总和替换SibSp和Parch，  
试着找出名字中与幸存属性相关的部分(例如，如果名字中包含“Countess”，那么幸存下来的可能性似乎更大)，  
尝试将数字属性转换为分类属性:例如，不同年龄组的存活率有很大差异(见下文)，所以创建一个年龄桶类别并使用它来代替年龄可能会有所帮助。 同样，为独自旅行的人设立一个特殊类别可能会很有用，因为只有30%的人幸存了下来(见下文)。  


```python
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>AgeBucket</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.576923</td>
    </tr>
    <tr>
      <th>15.0</th>
      <td>0.362745</td>
    </tr>
    <tr>
      <th>30.0</th>
      <td>0.423256</td>
    </tr>
    <tr>
      <th>45.0</th>
      <td>0.404494</td>
    </tr>
    <tr>
      <th>60.0</th>
      <td>0.240000</td>
    </tr>
    <tr>
      <th>75.0</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>RelativesOnboard</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.303538</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.552795</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.578431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.724138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.136364</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 垃圾邮件分类器


```python
"""
下载数据 
"""
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()
    print("下载完成!")

```


```python
fetch_spam_data()
```

    下载完成!
    


```python
"""
加载所有邮件
"""
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]
```


```python
len(ham_filenames)
```




    2500




```python
len(spam_filenames)
```




    500




```python
"""
我们可以使用 Python 的 email 模块来解析这些电子邮件（它处理标题、编码等）：
"""
import email
import email.policy

def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
```


```python
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
```


```python
"""
让我们看一个正常邮件的例子和一个垃圾邮件的例子，来感受一下数据的样子：
"""
print(ham_emails[1].get_content().strip())
```

    Martin A posted:
    Tassos Papadopoulos, the Greek sculptor behind the plan, judged that the
     limestone of Mount Kerdylio, 70 miles east of Salonika and not far from the
     Mount Athos monastic community, was ideal for the patriotic sculpture. 
     
     As well as Alexander's granite features, 240 ft high and 170 ft wide, a
     museum, a restored amphitheatre and car park for admiring crowds are
    planned
    ---------------------
    So is this mountain limestone or granite?
    If it's limestone, it'll weather pretty fast.
    
    ------------------------ Yahoo! Groups Sponsor ---------------------~-->
    4 DVDs Free +s&p Join Now
    http://us.click.yahoo.com/pt6YBB/NXiEAA/mG3HAA/7gSolB/TM
    ---------------------------------------------------------------------~->
    
    To unsubscribe from this group, send an email to:
    forteana-unsubscribe@egroups.com
    
     
    
    Your use of Yahoo! Groups is subject to http://docs.yahoo.com/info/terms/
    


```python
print(spam_emails[6].get_content().strip())
```

    Help wanted.  We are a 14 year old fortune 500 company, that is
    growing at a tremendous rate.  We are looking for individuals who
    want to work from home.
    
    This is an opportunity to make an excellent income.  No experience
    is required.  We will train you.
    
    So if you are looking to be employed from home with a career that has
    vast opportunities, then go:
    
    http://www.basetel.com/wealthnow
    
    We are looking for energetic and self motivated people.  If that is you
    than click on the link and fill out the form, and one of our
    employement specialist will contact you.
    
    To be removed from our link simple go to:
    
    http://www.basetel.com/remove.html
    
    
    4139vOLW7-758DoDY1425FRhM1-764SMFc8513fCsLl40
    


```python
"""
一些电子邮件实际上是多部分的，带有图像和附件（可以有自己的附件）。让我们看看我们拥有的各种类型的结构：
"""
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()
```


```python
from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures
```


```python
structures_counter(ham_emails).most_common()
```




    [('text/plain', 2408),
     ('multipart(text/plain, application/pgp-signature)', 66),
     ('multipart(text/plain, text/html)', 8),
     ('multipart(text/plain, text/plain)', 4),
     ('multipart(text/plain)', 3),
     ('multipart(text/plain, application/octet-stream)', 2),
     ('multipart(text/plain, text/enriched)', 1),
     ('multipart(text/plain, application/ms-tnef, text/plain)', 1),
     ('multipart(multipart(text/plain, text/plain, text/plain), application/pgp-signature)',
      1),
     ('multipart(text/plain, video/mng)', 1),
     ('multipart(text/plain, multipart(text/plain))', 1),
     ('multipart(text/plain, application/x-pkcs7-signature)', 1),
     ('multipart(text/plain, multipart(text/plain, text/plain), text/rfc822-headers)',
      1),
     ('multipart(text/plain, multipart(text/plain, text/plain), multipart(multipart(text/plain, application/x-pkcs7-signature)))',
      1),
     ('multipart(text/plain, application/x-java-applet)', 1)]




```python
structures_counter(spam_emails).most_common()
```




    [('text/plain', 218),
     ('text/html', 183),
     ('multipart(text/plain, text/html)', 45),
     ('multipart(text/html)', 20),
     ('multipart(text/plain)', 19),
     ('multipart(multipart(text/html))', 5),
     ('multipart(text/plain, image/jpeg)', 3),
     ('multipart(text/html, application/octet-stream)', 2),
     ('multipart(text/plain, application/octet-stream)', 1),
     ('multipart(text/html, text/plain)', 1),
     ('multipart(multipart(text/html), application/octet-stream, image/jpeg)', 1),
     ('multipart(multipart(text/plain, text/html), image/gif)', 1),
     ('multipart/alternative', 1)]



似乎正常电子邮件通常是纯文本，而垃圾邮件则有很多 HTML。此外，相当多的正常电子邮件是使用 PGP 签名的，而垃圾邮件没有。简而言之，电子邮件结构似乎是有用的信息。


```python
"""
现在让我们看一下电子邮件标题
"""

```




    '\n现在让我们看一下电子邮件标题\n'




```python
for header, value in spam_emails[0].items():
    print(header,":",value)
```

    Return-Path : <12a1mailbot1@web.de>
    Delivered-To : zzzz@localhost.spamassassin.taint.org
    Received : from localhost (localhost [127.0.0.1])	by phobos.labs.spamassassin.taint.org (Postfix) with ESMTP id 136B943C32	for <zzzz@localhost>; Thu, 22 Aug 2002 08:17:21 -0400 (EDT)
    Received : from mail.webnote.net [193.120.211.219]	by localhost with POP3 (fetchmail-5.9.0)	for zzzz@localhost (single-drop); Thu, 22 Aug 2002 13:17:21 +0100 (IST)
    Received : from dd_it7 ([210.97.77.167])	by webnote.net (8.9.3/8.9.3) with ESMTP id NAA04623	for <zzzz@spamassassin.taint.org>; Thu, 22 Aug 2002 13:09:41 +0100
    From : 12a1mailbot1@web.de
    Received : from r-smtp.korea.com - 203.122.2.197 by dd_it7  with Microsoft SMTPSVC(5.5.1775.675.6);	 Sat, 24 Aug 2002 09:42:10 +0900
    To : dcek1a1@netsgo.com
    Subject : Life Insurance - Why Pay More?
    Date : Wed, 21 Aug 2002 20:31:57 -1600
    MIME-Version : 1.0
    Message-ID : <0103c1042001882DD_IT7@dd_it7>
    Content-Type : text/html; charset="iso-8859-1"
    Content-Transfer-Encoding : quoted-printable
    


```python
"""
那里可能有很多有用的信息，例如发件人的电子邮件地址（12a1mailbot1@web.de 看起来很可疑），但我们只关注主题标题
"""
spam_emails[0]["Subject"]
```




    'Life Insurance - Why Pay More?'




```python
"""
好的，在我们对数据了解太多之前，我们不要忘记将其拆分为训练集和测试集
"""
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
"""
好的，让我们开始编写预处理函数。 首先，我们需要一个将 HTML 转换为纯文本的函数。 可以说，最好的方法是使用伟大的 BeautifulSoup
库，但我想避免向这个项目添加另一个依赖项，所以让我们使用正则表达式来破解一个快速而肮脏的解决方案（冒着 un̨ho͞ly radiańcé 
destro҉ying all enli̍̈́̂̈́ghtenment 的风险） ）。 下面的函数首先删除 <head> 部分，然后将所有 <a> 标签转换为单词 HYPERLINK，
然后去掉所有 HTML 标签，只留下纯文本。 为了可读性，它还用单个换行符替换了多个换行符，最后它对 html 实体（例如 &gt; 或 &nbsp;）
进行了转义：
"""
import re
from html import unescape

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

```


```python
"""
让我们看看它是否有效。这是 HTML 垃圾邮件
"""
html_spam_emails = [email for email in X_train[y_train==1]
                    if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")
```

    <HTML><HEAD><TITLE></TITLE><META http-equiv="Content-Type" content="text/html; charset=windows-1252"><STYLE>A:link {TEX-DECORATION: none}A:active {TEXT-DECORATION: none}A:visited {TEXT-DECORATION: none}A:hover {COLOR: #0033ff; TEXT-DECORATION: underline}</STYLE><META content="MSHTML 6.00.2713.1100" name="GENERATOR"></HEAD>
    <BODY text="#000000" vLink="#0033ff" link="#0033ff" bgColor="#CCCC99"><TABLE borderColor="#660000" cellSpacing="0" cellPadding="0" border="0" width="100%"><TR><TD bgColor="#CCCC99" valign="top" colspan="2" height="27">
    <font size="6" face="Arial, Helvetica, sans-serif" color="#660000">
    <b>OTC</b></font></TD></TR><TR><TD height="2" bgcolor="#6a694f">
    <font size="5" face="Times New Roman, Times, serif" color="#FFFFFF">
    <b>&nbsp;Newsletter</b></font></TD><TD height="2" bgcolor="#6a694f"><div align="right"><font color="#FFFFFF">
    <b>Discover Tomorrow's Winners&nbsp;</b></font></div></TD></TR><TR><TD height="25" colspan="2" bgcolor="#CCCC99"><table width="100%" border="0"  ...
    


```python
"""
这是由此产生的纯文本
"""
print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")
```

    
    OTC
     Newsletter
    Discover Tomorrow's Winners 
    For Immediate Release
    Cal-Bay (Stock Symbol: CBYI)
    Watch for analyst "Strong Buy Recommendations" and several advisory newsletters picking CBYI.  CBYI has filed to be traded on the OTCBB, share prices historically INCREASE when companies get listed on this larger trading exchange. CBYI is trading around 25 cents and should skyrocket to $2.66 - $3.25 a share in the near future.
    Put CBYI on your watch list, acquire a position TODAY.
    REASONS TO INVEST IN CBYI
    A profitable company and is on track to beat ALL earnings estimates!
    One of the FASTEST growing distributors in environmental & safety equipment instruments.
    Excellent management team, several EXCLUSIVE contracts.  IMPRESSIVE client list including the U.S. Air Force, Anheuser-Busch, Chevron Refining and Mitsubishi Heavy Industries, GE-Energy & Environmental Research.
    RAPIDLY GROWING INDUSTRY
    Industry revenues exceed $900 million, estimates indicate that there could be as much as $25 billi ...
    


```python
"""
太棒了！现在让我们编写一个函数，它将电子邮件作为输入并以纯文本形式返回其内容，无论其格式如何
"""
def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)
```


```python
print(email_to_text(sample_html_spam)[:100], "...")
```

    
    OTC
     Newsletter
    Discover Tomorrow's Winners 
    For Immediate Release
    Cal-Bay (Stock Symbol: CBYI)
    Wat ...
    


```python
"""
让我们加入一些词干吧！ 为此，您需要安装自然语言工具包 (NLTK)。 就像运行以下命令一样简单（不要忘记先激活您的 virtualenv；如果您没有，您可能需要管理员权限，或使用 --user 选项）
"""
try:
    import nltk

    stemmer = nltk.PorterStemmer()
    for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
        print(word, "=>", stemmer.stem(word))
except ImportError:
    print("Error: stemming requires the NLTK module.")
    stemmer = None
```

    Computations => comput
    Computation => comput
    Computing => comput
    Computed => comput
    Compute => comput
    Compulsive => compuls
    


```python
"""
我们还需要一种用“URL”一词替换 URL 的方法。 为此，我们可以使用核心正则表达式，但我们将只使用 urlextract 库。 您可以使用以下命令安装它（不要忘记先激活您的 virtualenv；如果您没有，您可能需要管理员权限，或使用 --user 选项）
"""
try:
    import urlextract # may require an Internet connection to download root domain names
    
    url_extractor = urlextract.URLExtract()
    print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None
```

    ['github.com', 'https://youtu.be/7Pq-S557XQU?t=3m32s']
    


```python
"""
我们已准备好将所有这些整合到一个转换器中，用于将电子邮件转换为文字计数器。 请注意，我们使用 Python 的 split() 
方法将句子拆分为单词，该方法使用空格作为单词边界。 这适用于许多书面语言，但不是全部。 
例如，中文和日文文字之间一般不使用空格，而越南文甚至在音节之间也经常使用空格。 
在这个练习中没问题，因为数据集（大部分）是英文的。
"""
from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)
```


```python
X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
X_few_wordcounts
```




    array([Counter({'chuck': 1, 'murcko': 1, 'wrote': 1, 'stuff': 1, 'yawn': 1, 'r': 1}),
           Counter({'the': 11, 'of': 9, 'and': 8, 'all': 3, 'christian': 3, 'to': 3, 'by': 3, 'jefferson': 2, 'i': 2, 'have': 2, 'superstit': 2, 'one': 2, 'on': 2, 'been': 2, 'ha': 2, 'half': 2, 'rogueri': 2, 'teach': 2, 'jesu': 2, 'some': 1, 'interest': 1, 'quot': 1, 'url': 1, 'thoma': 1, 'examin': 1, 'known': 1, 'word': 1, 'do': 1, 'not': 1, 'find': 1, 'in': 1, 'our': 1, 'particular': 1, 'redeem': 1, 'featur': 1, 'they': 1, 'are': 1, 'alik': 1, 'found': 1, 'fabl': 1, 'mytholog': 1, 'million': 1, 'innoc': 1, 'men': 1, 'women': 1, 'children': 1, 'sinc': 1, 'introduct': 1, 'burnt': 1, 'tortur': 1, 'fine': 1, 'imprison': 1, 'what': 1, 'effect': 1, 'thi': 1, 'coercion': 1, 'make': 1, 'world': 1, 'fool': 1, 'other': 1, 'hypocrit': 1, 'support': 1, 'error': 1, 'over': 1, 'earth': 1, 'six': 1, 'histor': 1, 'american': 1, 'john': 1, 'e': 1, 'remsburg': 1, 'letter': 1, 'william': 1, 'short': 1, 'again': 1, 'becom': 1, 'most': 1, 'pervert': 1, 'system': 1, 'that': 1, 'ever': 1, 'shone': 1, 'man': 1, 'absurd': 1, 'untruth': 1, 'were': 1, 'perpetr': 1, 'upon': 1, 'a': 1, 'larg': 1, 'band': 1, 'dupe': 1, 'import': 1, 'led': 1, 'paul': 1, 'first': 1, 'great': 1, 'corrupt': 1}),
           Counter({'url': 4, 's': 3, 'group': 3, 'to': 3, 'in': 2, 'forteana': 2, 'martin': 2, 'an': 2, 'and': 2, 'we': 2, 'is': 2, 'yahoo': 2, 'unsubscrib': 2, 'y': 1, 'adamson': 1, 'wrote': 1, 'for': 1, 'altern': 1, 'rather': 1, 'more': 1, 'factual': 1, 'base': 1, 'rundown': 1, 'on': 1, 'hamza': 1, 'career': 1, 'includ': 1, 'hi': 1, 'belief': 1, 'that': 1, 'all': 1, 'non': 1, 'muslim': 1, 'yemen': 1, 'should': 1, 'be': 1, 'murder': 1, 'outright': 1, 'know': 1, 'how': 1, 'unbias': 1, 'memri': 1, 'don': 1, 't': 1, 'html': 1, 'rob': 1, 'sponsor': 1, 'number': 1, 'dvd': 1, 'free': 1, 'p': 1, 'join': 1, 'now': 1, 'from': 1, 'thi': 1, 'send': 1, 'email': 1, 'egroup': 1, 'com': 1, 'your': 1, 'use': 1, 'of': 1, 'subject': 1})],
          dtype=object)




```python
"""
这看起来是对的！

现在我们有了字数，我们需要将它们转换为向量。 为此，我们将构建另一个转换器，其 fit() 方法将构建词汇表（最常见单词的有序列表），
其 transform() 方法将使用词汇表将词数转换为向量。 输出是一个稀疏矩阵。
"""
from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
```


```python
vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
X_few_vectors
```




    <3x11 sparse matrix of type '<class 'numpy.intc'>'
    	with 20 stored elements in Compressed Sparse Row format>




```python
X_few_vectors.toarray()
```




    array([[ 6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [99, 11,  9,  8,  3,  1,  3,  1,  3,  2,  3],
           [67,  0,  1,  2,  3,  4,  1,  2,  0,  1,  0]], dtype=int32)



这个矩阵是什么意思？ 那么，第二行第一列中的 99 表示第二封电子邮件包含 99 个不属于词汇表的单词。 旁边的 11 表示词汇表中的第一个单词在这封电子邮件中出现了 11 次。 旁边的 9 表示第二个词出现了 9 次，以此类推。 您可以查看词汇表以了解我们在谈论哪些单词。 第一个词是“the”，第二个词是“of”，以此类推。


```python
vocab_transformer.vocabulary_
```




    {'the': 1,
     'of': 2,
     'and': 3,
     'to': 4,
     'url': 5,
     'all': 6,
     'in': 7,
     'christian': 8,
     'on': 9,
     'by': 10}




```python
"""
我们现在准备训练我们的第一个垃圾邮件分类器！ 让我们转换整个数据集：
"""
from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
score.mean()
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    

    [CV] END ................................ score: (test=0.981) total time=   0.2s
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.3s remaining:    0.0s
    

    [CV] END ................................ score: (test=0.984) total time=   0.2s
    

    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.6s remaining:    0.0s
    

    [CV] END ................................ score: (test=0.990) total time=   0.4s
    

    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:    1.0s finished
    




    0.985



超过 98.5%，第一次尝试还不错！ :) 但是，请记住我们使用的是“简单”数据集。 您可以尝试使用更难的数据集，结果不会那么惊人。 您必须尝试多个模型，选择最好的模型并使用交叉验证对其进行微调，等等。

但是你得到了图片，所以让我们现在停止，只需打印出我们在测试集上获得的精度/召回率


```python
from sklearn.metrics import precision_score, recall_score

X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
log_clf.fit(X_train_transformed, y_train)

y_pred = log_clf.predict(X_test_transformed)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
```

    Precision: 96.88%
    Recall: 97.89%
    


```python

```
