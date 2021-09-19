---
layout: post
title: End-to-end-Machine-Learning-project
date: 2021-09-19 19:18
categories: [Hands-On Machine Learning]
tags: [regression,MachineLearning]
---
# 数据集

数据集来源于加州住房价格。该数据集基于1990年加州人口普查的数据。并且对于数据添加了一个分类属性，并且移除了一些特征。

![](/assets/images/img_HandsOnML/图2-1.jpg)

# 观察大局

**提出问题：** 使用加州人口普查数据建立起加州房价模型。数据中有许多指标，诸如每个街区人口数量，收入中位数，房价中位数等。

## 框架问题

模型的输出（对于一个房价中位数的预测）将会跟其他许多信号一起被传输到另一个机器学习系统。而这个下游系统将会被用来决策一个给定的区域是否值得投资。因为直接影响到收益，所以正确获得这个信息至关重要。


![image-20210730174142714](/assets/images/img_HandsOnML/图2-2.jpg)

在设计系统之前，还需要确定一些框架问题。首先，这是一个典型的**有监督学习** 任务，因为已经给出了训练事例（每个实例都有预期的产出，也就是该区域的中位数）。其次这是一个**回归任务** ，因为需要对于某个值进行预测。具体来说这是一个**多重回归问题** 因为系统要通过使用多个特征进行预测（使用区域人口，收入中位数等）。这也是一个**一元回归问题** 因为我们仅尝试预测每个区域的单个值。最后，我们没有一个连续的数据流不断流入系统，所以不需要针对变化的数据做出特别的调整，数据量也不是特别大，不需要特别多的内存，所以简单的批量学习就可以胜任。

## 选择性能指标

首先，回归问题的典型的性能指标是**均方根误差（RMSE）**。他给出了系统通常会在预测中产生多大误差，对于较大的误差，权重较高。

### 均方根误差（RMSE）

$$RMSE(X,h)=\sqrt{ {1\over m}{\sum_{i=1}^m(h(x^{(i)})-y^{(i)})^2} }$$

 **符号表示：**

- m是要在其上测量RMSE的数据集中实例数。

  - 例如，如果你要在2000个区域的验证集上评估RMSE,则 m=2000.

- $x^{(i)}$ 是数据集中第 i 个实例的所有特征值（不包含标签）的向量，而 y(i) 是其标签（该实例的期望输出值）。

  - 例如，如果数据集中的第一个区域位于经度 -118.29 ，维度 33.9，居民1416人，收入中位数为38372美元，房价中位数是156400美元（忽略其他特征）那么
  $$x^{(1)}=\begin{pmatrix}-118.29\\33.91\\1416\\38372\end{pmatrix}$$
  $$y^{(1)}=156400$$

- X 是一个矩阵，其中包含数据集中所有实例的所有特征值（不包含标签），每个实例仅有一行，第 *i* 行等于x(i)的转置。

  - 例如，如果第一个区域如上所述，则矩阵*X* 如下表示：

$$X=\begin{pmatrix}(x^{(1)})^T\\(x^{(2)})^T\\ \vdots \\(x^{(1999)})^T\\(x^{(2000)})^T\end{pmatrix}=\begin{pmatrix}-118.29 \ 33.91 \ 1416 \ 38372\\ \ \vdots \ \vdots \ \vdots \ \vdots \end{pmatrix}$$

- h 是系统的预测函数，也称为假设，当给系统输入一个实例的特征向量下x(i) 时，他会为该实例输出一个预测值 



  - 例如，如果系统预测第一个区域的房价中位数为 158400美元，则 $\hat y^{(i)}=h(x^{(i)})=158400$。该区域的预测误差为 $\hat y^{(1)}-y^{(1)}=2000$。

- RMSE（X,h) 是使用假设 h 在一组示例中测试的成本函数。

> 说明：一般情况下，我们将小写斜体字体用于标量值（例如 m或 y(i))和函数名称（例如 h),将小写粗斜体字体用于向量（例如 **x(i)**) ,将大写斜体用于矩阵表示（例如 **X**）。

尽管RMSE通常是回归任务的首选性能指标，但是在某些情况下，你可能更喜欢其他函数。例如假设有许多异常区域，在这种情况下，你可以考虑使用**平均绝对误差（Mean Absolute Error，MAE）**，也称为平均绝对误差。


### 平均绝对误差（MAE)

$$RMSE(X,h)={1\over m}{\sum_{i=1}^m|(h(x^{(i)})-y^{(i)})|}$$

RMSE 和MAE 都是测量两个向量（预测向量值和目标向量值）之间距离的方法。

# 获取数据

## 创建工作区

首先需要安装 [Python](https://www.python.org/) ,Python安装好之后还要安装一些Python模块：Jupter=，Numpy，Pandas ，Matplotlib 以及Scikit-Learn。**这里需要说明一点是：对于Python不同版本的模块可能是不兼容的，这也就意味着在安装Python模块时，需要相对应的模块版本，版本不兼容的话可能运行程序时会出现包无法导入的情况**。推荐版本如下：

```python
name: tf2
channels:
  - conda-forge
  - defaults
dependencies:
  - atari_py=0.2 # used only in chapter 18
  - box2d-py=2.3 # used only in chapter 18
  - ftfy=5.8 # used only in chapter 16 by the transformers library
  - graphviz # used only in chapter 6 for dot files
  - gym=0.18 # used only in chapter 18
  - ipython=7.20 # a powerful Python shell
  - ipywidgets=7.6 # optionally used only in chapter 12 for tqdm in Jupyter
  - joblib=0.14 # used only in chapter 2 to save/load Scikit-Learn models
  - jupyter=1.0 # to edit and run Jupyter notebooks
  - matplotlib=3.3 # beautiful plots. See tutorial tools_matplotlib.ipynb
  - nbdime=2.1 # optional tool to diff Jupyter notebooks
  - nltk=3.4 # optionally used in chapter 3, exercise 4
  - numexpr=2.7 # used only in the Pandas tutorial for numerical expressions
  - numpy=1.19 # Powerful n-dimensional arrays and numerical computing tools
  - opencv=4.5 # used only in chapter 18 by TF Agents for image preprocessing
  - pandas=1.2 # data analysis and manipulation tool
  - pillow=8.1 # image manipulation library, (used by matplotlib.image.imread)
  - pip # Python's package-management system
  - py-xgboost=0.90 # used only in chapter 7 for optimized Gradient Boosting
  - pyglet=1.5 # used only in chapter 18 to render environments
  - pyopengl=3.1 # used only in chapter 18 to render environments
  - python=3.7 # Python! Not using latest version as some libs lack support
  - python-graphviz # used only in chapter 6 for dot files
 #- pyvirtualdisplay=1.3 # used only in chapter 18 if on headless server
  - requests=2.25 # used only in chapter 19 for REST API queries
  - scikit-learn=0.24 # machine learning library
  - scipy=1.6 # scientific/technical computing library
  - tqdm=4.56 # a progress bar library
  - transformers=4.3 # Natural Language Processing lib for TF or PyTorch
  - wheel # built-package format for pip
  - widgetsnbextension=3.5 # interactive HTML widgets for Jupyter notebooks
  - pip:
    - tensorboard-plugin-profile==2.4.0 # profiling plugin for TensorBoard
    - tensorboard==2.4.1 # TensorFlow's visualization toolkit
    - tensorflow-addons==0.12.1 # used only in chapter 16 for a seq2seq impl.
    - tensorflow-datasets==3.0.0 # datasets repository, ready to use
    - tensorflow-hub==0.9.0 # trained ML models repository, ready to use
    - tensorflow-probability==0.12.1 # Optional. Probability/Stats lib.
    - tensorflow-serving-api==2.4.1 # or tensorflow-serving-api-gpu if gpu
    - tensorflow==2.4.2 # Deep Learning library
    - tf-agents==0.7.1 # Reinforcement Learning lib based on TensorFlow
    - tfx==0.27.0 # platform to deploy production ML pipelines
    - urlextract==1.2.0 # optionally used in chapter 3, exercise 4

```

Python 模块安装之后，现在就可以启动Jupyter进行下一步了。

## 下载数据


```python
# 下载数据脚本
'''
调用get_housing_data(),就会自动在工作区中创建一个 datasets/housing 目录，然后下载housing.tgz文件，并将housing.csv解压
到这个文件夹
'''
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
# 下载数据
'''
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # 递归创建多层目录。exist_ok:是否在目录存在时触发异常。False（默认）：触发异常FileExistsError。Ture：不触发异常
    os.makedirs(housing_path, exist_ok=True)
    # 连接两个或更多路径名。如果各组件名首字母不包含’/’，则函数会自动加上
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # 将URL表示的网络对象复制到本地
    urllib.request.urlretrieve(housing_url, tgz_path)
    # 解压缩一个tar包
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
    print("下载完毕！！")
    
fetch_housing_data()
'''
print("下载完毕！！")

```

    下载完毕！！


> 说明：由于数据服务器在国外，又由于某些客观的原因，这里下载数据需要**科学上网**。需要自行解决。


```python
# 使用pandas加载数据
'''
函数会返回一个包含所有数据得pandas DataFrame 对象
'''
import pandas as pd
import numpy as np

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()

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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>



## 快速查看数据结构

使用pandas加载数据


```python
# 查看数据集中的前五行
housing.head()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 查看数据集的简单描述，特别是总行数，每个属性类型和非空值的数量
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB



```python
# 查看有多少种分类存在
housing["ocean_proximity"].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64




```python
# 显示数值属性摘要
housing.describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 绘制直方图
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50 ,figsize=(20,15))
plt.show()
```


​    
![png](/assets/images/img_HandsOnML/output_28_0.png)
​    


## 创建测试集


```python
# 创建测试集--数据集的20%
import numpy as np
def split_train_test(data,test_ratio):
    # 设置随机数生成器种子
    np.random.seed(42)
    # #permutation随机生成0-len(data)随机序列
    shuffled_indices=np.random.permutation(len(data))
    #test_ratio为测试集所占的百分比
    test_set_size = int(len(data)*test_ratio)
    #前面的test_set_size个为测试集
    test_indices = shuffled_indices[:test_set_size]
    #后面的test_set_size个为训练集
    train_indices = shuffled_indices[test_set_size:]
    #iloc选择参数序列中所对应的行
    return data.iloc[train_indices],data.iloc[test_indices]

train_set, test_set = split_train_test(housing,0.2)
print(len(train_set))
print(train_set)

print(len(test_set))
print(test_set)

```

    16512
           longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
    14196    -117.03     32.71                33.0       3126.0           627.0   
    8267     -118.16     33.77                49.0       3382.0           787.0   
    17445    -120.48     34.66                 4.0       1897.0           331.0   
    14265    -117.11     32.69                36.0       1421.0           367.0   
    2271     -119.80     36.78                43.0       2382.0           431.0   
    ...          ...       ...                 ...          ...             ...   
    11284    -117.96     33.78                35.0       1330.0           201.0   
    11964    -117.43     34.02                33.0       3084.0           570.0   
    5390     -118.38     34.03                36.0       2101.0           569.0   
    860      -121.96     37.58                15.0       3575.0           597.0   
    15795    -122.42     37.77                52.0       4226.0          1315.0   
    
           population  households  median_income  median_house_value  \
    14196      2300.0       623.0         3.2596            103000.0   
    8267       1314.0       756.0         3.8125            382100.0   
    17445       915.0       336.0         4.1563            172600.0   
    14265      1418.0       355.0         1.9425             93400.0   
    2271        874.0       380.0         3.5542             96500.0   
    ...           ...         ...            ...                 ...   
    11284       658.0       217.0         6.3700            229200.0   
    11964      1753.0       449.0         3.0500             97800.0   
    5390       1756.0       527.0         2.9344            222100.0   
    860        1777.0       559.0         5.7192            283500.0   
    15795      2619.0      1242.0         2.5755            325000.0   
    
          ocean_proximity  
    14196      NEAR OCEAN  
    8267       NEAR OCEAN  
    17445      NEAR OCEAN  
    14265      NEAR OCEAN  
    2271           INLAND  
    ...               ...  
    11284       <1H OCEAN  
    11964          INLAND  
    5390        <1H OCEAN  
    860         <1H OCEAN  
    15795        NEAR BAY  
    
    [16512 rows x 10 columns]
    4128
           longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
    20046    -119.01     36.06                25.0       1505.0             NaN   
    3024     -119.46     35.14                30.0       2943.0             NaN   
    15663    -122.44     37.80                52.0       3830.0             NaN   
    20484    -118.72     34.28                17.0       3051.0             NaN   
    9814     -121.93     36.62                34.0       2351.0             NaN   
    ...          ...       ...                 ...          ...             ...   
    15362    -117.22     33.36                16.0       3165.0           482.0   
    16623    -120.83     35.36                28.0       4323.0           886.0   
    18086    -122.05     37.31                25.0       4111.0           538.0   
    2144     -119.76     36.77                36.0       2507.0           466.0   
    3665     -118.37     34.22                17.0       1787.0           463.0   
    
           population  households  median_income  median_house_value  \
    20046      1392.0       359.0         1.6812             47700.0   
    3024       1565.0       584.0         2.5313             45800.0   
    15663      1310.0       963.0         3.4801            500001.0   
    20484      1705.0       495.0         5.7376            218600.0   
    9814       1063.0       428.0         3.7250            278000.0   
    ...           ...         ...            ...                 ...   
    15362      1351.0       452.0         4.6050            263300.0   
    16623      1650.0       705.0         2.7266            266800.0   
    18086      1585.0       568.0         9.2298            500001.0   
    2144       1227.0       474.0         2.7850             72300.0   
    3665       1671.0       448.0         3.5521            151500.0   
    
          ocean_proximity  
    20046          INLAND  
    3024           INLAND  
    15663        NEAR BAY  
    20484       <1H OCEAN  
    9814       NEAR OCEAN  
    ...               ...  
    15362       <1H OCEAN  
    16623      NEAR OCEAN  
    18086       <1H OCEAN  
    2144           INLAND  
    3665        <1H OCEAN  
    
    [4128 rows x 10 columns]



```python
# 每一个实例都使用一个标识符来决定是否进入测试集
'''
计算每一个实例标识符的哈希值，如果这个哈希值小于或者等于最大哈希值(2**32)的20%，则将该实例放入测试集。
这样可以保证测试集在多个运行里都是一致的，即使是新加入的数据集也仍然一致。
新实例的20%将会被放入新的测试集，而之前训练集的实例也不会被放入新的测试集
'''
from zlib import crc32
import numpy as np
def test_set_check(identifier ,test_ratio):
    '''
    zlib.crc32(data[, value])
    crc32用于计算 data 的 CRC (循环冗余校验) 值。计算的结果是一个 32 位的整数。
    参数 value 是校验时的起始值，其默认值为 0。借助参数 value 可为分段的输入计算校验值。
    此算法没有加密强度，不应用于身份验证和数字签名。此算法的目的仅为验证数据的正确性，不适合作为通用散列算法。
    
    在python 3.0 之后: 返回值永远是无符号数。要在所有的 Python 版本和平台上获得相同的值，
    请使用 crc32(data) & 0xffffffff。
    '''
    # 是否小于最大索引的 20% ，返回 True or False 
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data,test_ratio,id_column):
    # 获取id列
    ids=data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_ ,test_ratio))
    # print(in_test_set)
    '''
    0        False
    1        False
    2         True
    3        False
    4        False
         ...  
    20635    False
    20636    False
    20637    False
    20638    False
    20639    False
    Name: index, Length: 20640, dtype: bool
    '''
    return data.loc[~in_test_set],data.loc[in_test_set]

# 添加一个index列
housing_with_id =housing.reset_index()
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"index")

# 取经纬度作为 index 
housing_with_id["id"]=housing["longitude"]*1000+housing["latitude"]
train_set, test_set =split_train_test_by_id(housing_with_id,0.2,"id")
```


```python
# Scikit-Learn 函数划分数据集
from sklearn.model_selection import train_test_split
train_set ,test_set =train_test_split(housing,test_size=0.2,random_state=42)
print(len(train_set))
print(len(test_set))
```

    16512
    4128



```python
# 使用 pd.cut() 来创建5个收入类别属性的（用1~5）来做标签，0~1.5 是类别 1 ,1.5~3 是类别 2

import numpy as np
housing = load_housing_data()
housing["income_cat"]= pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])
# housing["income_cat"].hist()

# 分层抽样 Scikit-Learn StratifiedShuffle-Split
from sklearn.model_selection import StratifiedShuffleSplit

# 参数 n_splits是将训练数据分成train/test对的组数，可根据需要进行设置，默认为10
split = StratifiedShuffleSplit(n_splits=1 ,test_size=0.2 ,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]
    

start_train_set["income_cat"].value_counts() / len(start_train_set)

start_test_set["income_cat"].value_counts() / len(start_test_set)


#删除 income_cat 属性
for set_ in (start_train_set,start_test_set):
    set_.drop("income_cat",axis=1,inplace=True)

print(start_train_set)
print(start_test_set)
```

           longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
    17606    -121.89     37.29                38.0       1568.0           351.0   
    18632    -121.93     37.05                14.0        679.0           108.0   
    14650    -117.20     32.77                31.0       1952.0           471.0   
    3230     -119.61     36.31                25.0       1847.0           371.0   
    3555     -118.59     34.23                17.0       6592.0          1525.0   
    ...          ...       ...                 ...          ...             ...   
    6563     -118.13     34.20                46.0       1271.0           236.0   
    12053    -117.56     33.88                40.0       1196.0           294.0   
    13908    -116.40     34.09                 9.0       4855.0           872.0   
    11159    -118.01     33.82                31.0       1960.0           380.0   
    15775    -122.45     37.77                52.0       3095.0           682.0   
    
           population  households  median_income  median_house_value  \
    17606       710.0       339.0         2.7042            286600.0   
    18632       306.0       113.0         6.4214            340600.0   
    14650       936.0       462.0         2.8621            196900.0   
    3230       1460.0       353.0         1.8839             46300.0   
    3555       4459.0      1463.0         3.0347            254500.0   
    ...           ...         ...            ...                 ...   
    6563        573.0       210.0         4.9312            240200.0   
    12053      1052.0       258.0         2.0682            113000.0   
    13908      2098.0       765.0         3.2723             97800.0   
    11159      1356.0       356.0         4.0625            225900.0   
    15775      1269.0       639.0         3.5750            500001.0   
    
          ocean_proximity  
    17606       <1H OCEAN  
    18632       <1H OCEAN  
    14650      NEAR OCEAN  
    3230           INLAND  
    3555        <1H OCEAN  
    ...               ...  
    6563           INLAND  
    12053          INLAND  
    13908          INLAND  
    11159       <1H OCEAN  
    15775        NEAR BAY  
    
    [16512 rows x 10 columns]
           longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
    5241     -118.39     34.12                29.0       6447.0          1012.0   
    10970    -117.86     33.77                39.0       4159.0           655.0   
    20351    -119.05     34.21                27.0       4357.0           926.0   
    6568     -118.15     34.20                52.0       1786.0           306.0   
    13285    -117.68     34.07                32.0       1775.0           314.0   
    ...          ...       ...                 ...          ...             ...   
    20519    -121.53     38.58                33.0       4988.0          1169.0   
    17430    -120.44     34.65                30.0       2265.0           512.0   
    4019     -118.49     34.18                31.0       3073.0           674.0   
    12107    -117.32     33.99                27.0       5464.0           850.0   
    2398     -118.91     36.79                19.0       1616.0           324.0   
    
           population  households  median_income  median_house_value  \
    5241       2184.0       960.0         8.2816            500001.0   
    10970      1669.0       651.0         4.6111            240300.0   
    20351      2110.0       876.0         3.0119            218200.0   
    6568       1018.0       322.0         4.1518            182100.0   
    13285      1067.0       302.0         4.0375            121300.0   
    ...           ...         ...            ...                 ...   
    20519      2414.0      1075.0         1.9728             76400.0   
    17430      1402.0       471.0         1.9750            134000.0   
    4019       1486.0       684.0         4.8984            311700.0   
    12107      2400.0       836.0         4.7110            133500.0   
    2398        187.0        80.0         3.7857             78600.0   
    
          ocean_proximity  
    5241        <1H OCEAN  
    10970       <1H OCEAN  
    20351       <1H OCEAN  
    6568           INLAND  
    13285          INLAND  
    ...               ...  
    20519          INLAND  
    17430      NEAR OCEAN  
    4019        <1H OCEAN  
    12107          INLAND  
    2398           INLAND  
    
    [4128 rows x 10 columns]


# 从数据探索和可视化中获得洞见

## 将地理数据可视化


```python
housing =start_train_set.copy()
housing.plot(kind="scatter",x="longitude",y="latitude")

```




    <AxesSubplot:xlabel='longitude', ylabel='latitude'>




​    
![png](/assets/images/img_HandsOnML/output_36_1.png)
​    



```python
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
```




    <AxesSubplot:xlabel='longitude', ylabel='latitude'>




​    
![png](/assets/images/img_HandsOnML/output_37_1.png)
​    

```python
# 查看房价
import matplotlib.pyplot as plt
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
```




    <matplotlib.legend.Legend at 0x202b7bad0c8>




​    
![png](/assets/images/img_HandsOnML/output_38_1.png)
​    


## 寻找相关性


```python
# 使用corr()方法计算每对属性之间的相关系数
corr_matrix=housing.corr()
```


```python
# 查看每个属性和房价中位数的相关系数
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.687160
    total_rooms           0.135097
    housing_median_age    0.114110
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724
    Name: median_house_value, dtype: float64




```python
# 使用pandas 的scatter_matrix函数来检测属性之间的相关性
from pandas.plotting import scatter_matrix

attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))
```




    array([[<AxesSubplot:xlabel='median_house_value', ylabel='median_house_value'>,
            <AxesSubplot:xlabel='median_income', ylabel='median_house_value'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='median_house_value'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='median_house_value'>],
           [<AxesSubplot:xlabel='median_house_value', ylabel='median_income'>,
            <AxesSubplot:xlabel='median_income', ylabel='median_income'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='median_income'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='median_income'>],
           [<AxesSubplot:xlabel='median_house_value', ylabel='total_rooms'>,
            <AxesSubplot:xlabel='median_income', ylabel='total_rooms'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='total_rooms'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='total_rooms'>],
           [<AxesSubplot:xlabel='median_house_value', ylabel='housing_median_age'>,
            <AxesSubplot:xlabel='median_income', ylabel='housing_median_age'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='housing_median_age'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='housing_median_age'>]],
          dtype=object)




​    
![png](/assets/images/img_HandsOnML/output_42_1.png)
​    



```python
# 收入中位数与房价中位数
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
```




    <AxesSubplot:xlabel='median_income', ylabel='median_house_value'>




​    
![png](/assets/images/img_HandsOnML/output_43_1.png)
​    


## 实验不同属性的组合


```python
# 创建一些新的属性，rooms_per_household,bedrooms_per_room,poputation_per_household
housing["room_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```


```python
# 相关系数
corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.687160
    room_per_household          0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64



# 机器学习算法准备


```python
# 先回到一个干净的数据集（再次复制start_train_set),然后将预测器与标签分开。drop() :创建一个副本，但是不影响start_train_set
housing =start_train_set.drop("median_house_value",axis=1) # 原始数据并未发生改变
housing_labels =start_train_set["median_house_value"].copy()
```

## 数据清理

total_bedrooms属性有部分值缺失，因此需要解决。以下有三种方法
1. 放弃这些相应的区域
2. 放弃整个属性
3. 将缺失的值设置为某个值（0，平均数或中位数）


```python
# 通过DataFrame的 dropna(),drop()和fillna() 可以实现上述操作
"""
housing.dropna(subset=["total_bedrooms"]) # option1
housing.drop("total_bedrooms",axis=1) # option2
median=housing["total_bedrooms"].median()
housing.["total_bedrooms"].fillna(median,inplace=True) # option3
"""
```




    '\nhousing.dropna(subset=["total_bedrooms"]) # option1\nhousing.drop("total_bedrooms",axis=1) # option2\nmedian=housing["total_bedrooms"].median()\nhousing.["total_bedrooms"].fillna(median,inplace=True) # option3\n'



Scikit-learn 提供了一个容易上手的类来处理确缺失值：SimpleImputer


```python
# 缺失处理
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# 由于中位数只能在数值属性上计算，所以需要创建一个没有文本属性 ocean_proximity的数据副本
housing_num=housing.drop("ocean_proximity",axis=1)
# 使用 fit() 方法将imputer 实例适配到训练数据
imputer.fit(housing_num)
```




    SimpleImputer(strategy='median')



这里imputer仅仅只是计算了每个属性的中位数，并将结果存储到statistics_中。虽然现在仅有total_bedrooms这个属性存在缺失，
但是我们无法确定系统启动之后新数据是否一定不存在任何缺失值，为了稳妥起见，将imputer应用于所有的数据属性


```python
imputer.statistics_
```




    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409])




```python
housing_num.median().values
```




    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409])




```python
# 使用Imputer 将缺失值替换成中位数
X=imputer.transform(housing_num)
```

结果是一个包含转换后特征的Numpy数组。


```python
# 将转换后数据放回pandas DataFrame 
housing_tr=pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
```

## 处理文本和分类属性

到目前为止，我们只处理数值属性，但是还要处理文本属性。在此数据集中仅有一个文本属性 ：ocean_proximity


```python
# 观察一下前十个实例的值
housing_cat=housing[["ocean_proximity"]]
housing_cat.head(10)
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
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>19480</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>8879</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13685</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>4937</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



ocean_proximity 属性类型是Object。使用Value——counts() 查看ocean_proximity属性


```python
housing["ocean_proximity"].value_counts()
```




    <1H OCEAN     7276
    INLAND        5263
    NEAR OCEAN    2124
    NEAR BAY      1847
    ISLAND           2
    Name: ocean_proximity, dtype: int64



ocean_proximity属性的值不是任意文本，而是有限的可能的取值，每一个只值代表一个类别。因此此属性是一个分类属性。为了适应机器学习算法，
因此我们将文本转换成数字。这里可以使用Scikit-Learn的OrdinalEncoder类


```python
# 将 ocean_proximity属性的值，由文本更改为数字
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```




    array([[0.],
           [0.],
           [4.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.]])



可以使用Categories_实例变量获取类别列表


```python
ordinal_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]




```python
# Scikit-Learn 提供了一个 OneHotEncoder 编码器，可以将整数类型转换成独热向量。
from  sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>



> 这里输出的是一个Scipy稀疏矩阵。如果想把他转换成一个Numpy数组，只需要调用toarray()


```python
housing_cat_1hot.toarray()
```




    array([[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           ...,
           [0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.]])




```python
cat_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



## 自定义转换器


```python
# 自定义转换器
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

## 特征缩放

1. 最小-最大缩放（归一化）
    将值重新缩放使其最终范围归于0~1之间。实现方法是将值减去最小值除以最大值与最小值的差。对此Scikit-Learn 提供了一个名为MinMaxScaler 的转换器。
2. 标准化
    首先减去平均值，然后除以方差。从而使的结果的分布具备单位方差。Scikit-Learn提供了一个标准化的转换器 StandadScaler 

## 转换流水线

正如你所见，许多数据的转换的步骤需要以正确的顺序来执行。而Scikit-Learn 提供了Pipeline类来支持这样的转换。


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")), # 中值写入
    ('attribs_adder',CombinedAttributesAdder()), # 自定义的转换器
    ('std_scaler',StandardScaler()) # 标准化
])
housing_num_tr=num_pipeline.fit_transform(housing_num)

```


```python
# 将所有转换应用到房屋数据
from sklearn.compose import ColumnTransformer

num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

full_pipeline= ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(),cat_attribs), # 编码器 编码成独热向量
])
housing_prepared=full_pipeline.fit_transform(housing)
```

# 选择和训练模型

## 训练和评估训练集


```python
# 训练一个线性回归模型
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
```




    LinearRegression()




```python
# 使用训练集
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
some_data_prepared=full_pipeline.transform(some_data)
print("Predictions:",lin_reg.predict(some_data_prepared))
print("Labels:",list(some_labels))
```

    Predictions: [210644.60459286 317768.80697211 210956.43331178  59218.98886849
     189747.55849879]
    Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]



```python
# 使用Scikit-Learn的 mean_squared_error() 函数来测量整个训练集上回归模型的RMSE（均方根误差）
from sklearn.metrics import mean_squared_error

housing_predictions=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse
```




    68628.19819848922



多数区域的 median_housing_values 分布在120000~265000之间，而典型的预测误差达到 68628 美元，这个预测结果显然的差强人意的。这是一个典型的模型对训练数据欠拟合的案例。
想要修正欠拟合问题，只有使用更强大的模型或者为算法训练提供更好的特征，又或者减少对模型的限制。由于我们这个模型不是个**正则化**的模型，所以排除了最后那个选项。因此可以添加更多的属性（例如，人口数量日志等），但是首先我们尝试一个更加复杂的模型。
我们训练一个**DecisionTreeRegressor（决策树）**。这是一个非常强大模型，能够从数据中找到复杂的非线性关系。


```python
# 决策树模型训练
from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
```




    DecisionTreeRegressor()




```python
# 训练集评估模型
housing_predictions=tree_reg.predict(housing_prepared)
tree_mse=mean_squared_error(housing_labels,housing_predictions)
tree_rmse=np.sqrt(tree_mse)
tree_rmse
```




    0.0



## 使用交叉验证来更好的进行评估


```python
# K-折交叉验证功能
"""
他将训练及随机分成10个不同的子集，每个子集称为一个叠加，然后对决策树模型进行10次训练和评估——每次挑选一个叠加进行评估，使用另外九个
折叠进行训练。产生结果是一个包含10次评估的数组。
"""
from sklearn.model_selection import cross_val_score

scores=cross_val_score(tree_reg,housing_prepared,housing_labels,
                      scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores=np.sqrt(-scores)
```

> Scikit-Learn 的交叉验证功能更倾向于使用有效函数（越大越好）而不是成本函数（越小越好）所以计算分数的函数实际是输出的一个负的MSE,这也是为什么计算平方根时是负的


```python
# 查看结果
def display_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())
    
display_scores(tree_rmse_scores)
```

    Scores: [69327.01708558 65486.39211857 71358.25563341 69091.37509104
     70570.20267046 75529.94622521 69895.20650652 70660.14247357
     75843.74719231 68905.17669382]
    Mean: 70666.74616904806
    Standard deviation: 2928.322738055112



```python
# 线性回归模型评分
line_scores=cross_val_score(lin_reg,housing_prepared,housing_labels,
                           scoring="neg_mean_squared_error",cv=10)
lin_rmse_scores=np.sqrt(-line_scores)
display_scores(lin_rmse_scores)
```

    Scores: [66782.73843989 66960.118071   70347.95244419 74739.57052552
     68031.13388938 71193.84183426 64969.63056405 68281.61137997
     71552.91566558 67665.10082067]
    Mean: 69052.46136345083
    Standard deviation: 2731.674001798347


通过比较可以看出线性回归模型的评分要优于决策树模型，这也就意味着决策树模型严重过拟合。
接下来看一下最后一个模型RandomForestRegressor(随机森林）


```python
# RandomForestRegressor（随机森林）
'''
随机森林简单原理描述：通过对特征的随机子集进行许多个决策树训练，然后对其预测取平均值。再多个模型基础之上建立模型，
称之为集成学习，这是进一步推动了机器学习算法的好方法。
'''
from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)
```




    RandomForestRegressor()




```python
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
```




    18680.294240259147




```python
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
```

    Scores: [49557.6095063  47584.54435547 49605.349788   52325.13724488
     49586.9889247  53154.87424699 48800.48987508 47880.32844243
     52958.68645964 50046.17489414]
    Mean: 50150.018373763225
    Standard deviation: 1902.0697041387534



```python
'''
# 保存Scikit-Learn模型
import joblib

joblib.dump(my_model,"my_model.pkl")
# Later 
my_model_loaded=joblib.load("my_model.pkl")
'''
```




    '\n# 保存Scikit-Learn模型\nimport joblib\n\njoblib.dump(my_model,"my_model.pkl")\n# Later \nmy_model_loaded=joblib.load("my_model.pkl")\n'



# 微调模型

经过各种模型（线性回归，决策树，随机森林...)的探索，现在有了一个有效模型的候选列表。现在需要对于这些模型进行微调。

## 网格搜索

一种微调的方法是手动调整参数，知道找到一组很好的超参数类型。Scikit-Learn的 GridSearchCV 可以自动进行探索。


```python
# 搜索RandomForestRegressor的超参数值得最佳组合
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```




    GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]},
                             {'bootstrap': [False], 'max_features': [2, 3, 4],
                              'n_estimators': [3, 10]}],
                 return_train_score=True, scoring='neg_mean_squared_error')



> 说明：para_grid 告诉Scikit-Learn，首先评估第一个dict中的 n_estimator 和 max_feactures 的所有3*4=12 种超参数组合（这些超参数将在第七章进行解释）；接着尝试第二个dict中的超参数2*3=6种组合，这次超参数bootstrap需要设置为False而不是Ture（Ture是该超参数的默认值）。

总而言之，网格搜索将探索RandomForestRegressor超参数值得12+6=18种组合，并对每个模型进行5次训练（因为我们使用的是**5-折交叉验证**


```python
# 获取最佳的参数组合
grid_search.best_params_
```




    {'max_features': 8, 'n_estimators': 30}




```python
# 得到最好的估算器
grid_search.best_estimator_
```




    RandomForestRegressor(max_features=8, n_estimators=30, random_state=42)




```python
# 评估分数
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)
```

    63669.11631261028 {'max_features': 2, 'n_estimators': 3}
    55627.099719926795 {'max_features': 2, 'n_estimators': 10}
    53384.57275149205 {'max_features': 2, 'n_estimators': 30}
    60965.950449450494 {'max_features': 4, 'n_estimators': 3}
    52741.04704299915 {'max_features': 4, 'n_estimators': 10}
    50377.40461678399 {'max_features': 4, 'n_estimators': 30}
    58663.93866579625 {'max_features': 6, 'n_estimators': 3}
    52006.19873526564 {'max_features': 6, 'n_estimators': 10}
    50146.51167415009 {'max_features': 6, 'n_estimators': 30}
    57869.25276169646 {'max_features': 8, 'n_estimators': 3}
    51711.127883959234 {'max_features': 8, 'n_estimators': 10}
    49682.273345071546 {'max_features': 8, 'n_estimators': 30}
    62895.06951262424 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    54658.176157539405 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    59470.40652318466 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    52724.9822587892 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    57490.5691951261 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51009.495668875716 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}


> 从输出可以得到最佳的解决方案是将超参数 max_features设置为8 ，将超参数n_estimators 设置为30 。这个组合的RNSE分数为 49682 。

## 随机搜索

如果组合数量较少，那么网格搜索是一个不错的选择。但是当超参数搜索范围较大时，通常优先使用**RandomizedSearchCV** 相比较而言，这个类不会计算所有的组合，而是在每次的迭代中为每个超参数选择一个随机值然后对对一定数量的随机组合进行评估。

**优势**
- 如果运行随机搜索1000个迭代，那么将会探索每个超参数的1000个不同的取值（而不是网格搜索那样每个超参数仅仅探索少量的几个值。
- 通过简单的设置迭代次数，可以更好的控制要分配置给超参数搜索的计算预算。

## 集成方法

将表现最优的模型组合起来。

## 分析最佳模型极其误差


```python
# 指出每个属性的相对重要程度
feature_importances=grid_search.best_estimator_.feature_importances_
feature_importances
```




    array([7.33442355e-02, 6.29090705e-02, 4.11437985e-02, 1.46726854e-02,
           1.41064835e-02, 1.48742809e-02, 1.42575993e-02, 3.66158981e-01,
           5.64191792e-02, 1.08792957e-01, 5.33510773e-02, 1.03114883e-02,
           1.64780994e-01, 6.02803867e-05, 1.96041560e-03, 2.85647464e-03])




```python
# 将这些重要重要性的分数显示在对应的属性旁边
extra_attribs=["rooms_per_hhold","pop_per_hhold","bedrooms_per_room"]
cat_encoder=full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs=list(cat_encoder.categories_[0])
attributes=num_attribs+extra_attribs+cat_one_hot_attribs
sorted(zip(feature_importances,attributes),reverse=True)
```




    [(0.36615898061813423, 'median_income'),
     (0.16478099356159054, 'INLAND'),
     (0.10879295677551575, 'pop_per_hhold'),
     (0.07334423551601243, 'longitude'),
     (0.06290907048262032, 'latitude'),
     (0.056419179181954014, 'rooms_per_hhold'),
     (0.053351077347675815, 'bedrooms_per_room'),
     (0.04114379847872964, 'housing_median_age'),
     (0.014874280890402769, 'population'),
     (0.014672685420543239, 'total_rooms'),
     (0.014257599323407808, 'households'),
     (0.014106483453584104, 'total_bedrooms'),
     (0.010311488326303788, '<1H OCEAN'),
     (0.0028564746373201584, 'NEAR OCEAN'),
     (0.0019604155994780706, 'NEAR BAY'),
     (6.0280386727366e-05, 'ISLAND')]



## 通过测试集评估系统

现在已经获得了一个足够优秀的的系统，现在需要使用测试集来评估最终的模型。

只需要从测试集中获取预测器和标签，运行full_pipeline来数据转换（调用transfrom()而不是fit_transfrom()),然后在测试集上评估模型：


```python
final_model = grid_search.best_estimator_

X_test = start_test_set.drop("median_house_value", axis=1)
y_test = start_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```


```python
# 使用scipy.stats.t.interval()计算泛化误差的95%置信区间
from scipy import stats
confidence=0.95
squared_errors=(final_predictions-y_test)**2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
```




    array([45685.10470776, 49691.25001878])



# 启动，监控，维护系统


```python
# 
```
