[TOC]

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
%matplotlib inline
```


```python
plt.rcParams['font.family'] = 'SimHei' # 全局设置为黑体
plt.rcParams['font.size'] = 15
```

# 1 数据处理

## 1.1 预处理：观察


```python
shfj = pd.read_csv('house.csv',index_col=0,na_values=['暂无资料','暂无资料年'])#读取，并指定索引、缺失值
```


```python
shfj.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 240 entries, 0 to 239
    Data columns (total 14 columns):
    標題       240 non-null object
    产权性质     228 non-null object
    住宅类别     218 non-null object
    建筑类别     188 non-null object
    参考月供     0 non-null float64
    年代       223 non-null object
    建筑面积     240 non-null float64
    户型       240 non-null object
    楼层       228 non-null object
    物 业 费    215 non-null object
    物业类型     240 non-null object
    结构       181 non-null object
    装修       228 non-null object
    總價       237 non-null float64
    dtypes: float64(3), object(11)
    memory usage: 17.8+ KB



```python
shfj.head(1)
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
      <th>標題</th>
      <th>产权性质</th>
      <th>住宅类别</th>
      <th>建筑类别</th>
      <th>参考月供</th>
      <th>年代</th>
      <th>建筑面积</th>
      <th>户型</th>
      <th>楼层</th>
      <th>物 业 费</th>
      <th>物业类型</th>
      <th>结构</th>
      <th>装修</th>
      <th>總價</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>急售70年产权房源 房东装修 户型正气 产证清晰 随时看房</td>
      <td>个人产权</td>
      <td>普通住宅</td>
      <td>板楼</td>
      <td>NaN</td>
      <td>1989年</td>
      <td>70.4</td>
      <td>2室2厅1厨1卫</td>
      <td>中层(共6层)</td>
      <td>NaN</td>
      <td>住宅</td>
      <td>平层</td>
      <td>精装修</td>
      <td>30.6</td>
    </tr>
  </tbody>
</table>
</div>



## 1.2 缺失值处理
* 缺失值统计
* 根据需求，舍弃或填充缺失值
* 拆分字符串列


```python
shfj.isnull().sum() # 缺失值统计
```




    標題         0
    产权性质      12
    住宅类别      22
    建筑类别      52
    参考月供     240
    年代        17
    建筑面积       0
    户型         0
    楼层        12
    物 业 费     25
    物业类型       0
    结构        59
    装修        12
    總價         3
    dtype: int64




```python
shfj.isnull().sum()/shfj.count()# 统计各列缺失值占比
```




    標題       0.000000
    产权性质     0.052632
    住宅类别     0.100917
    建筑类别     0.276596
    参考月供          inf
    年代       0.076233
    建筑面积     0.000000
    户型       0.000000
    楼层       0.052632
    物 业 费    0.116279
    物业类型     0.000000
    结构       0.325967
    装修       0.052632
    總價       0.012658
    dtype: float64




```python
shfj.drop('参考月供',axis=1,inplace=True) # 删除全为缺失值的列
```

###  1.2.1 【物业费】 列的处理
* 找规律，整理为单一数字
* 填充缺失值


```python
(shfj['物 业 费'].dropna().str.contains('元/')).sum()# 寻找列中各元素 包含的 共同特征
```




    215




```python
shfj['物业费（元/平米）']=shfj['物 业 费'].str.split('元',expand=True)[0].astype(float)# 只取数组，创建新列
```


```python
shfj['物业费（元/平米）'].fillna(shfj['物业费（元/平米）'].median(),inplace=True)# 缺失值填充为该列 中位数
```

### 1.2.2 【總價】 列的处理
* 缺失值填充为：每平米均价 * 建筑面积
* 0~1标准化


```python
shfj['总价（万元）']=shfj['總價'].fillna((shfj['總價']/shfj['建筑面积']).mean()*shfj['建筑面积'])# 填充缺失值=每平米价格的平均值 * 建筑面积
```


```python
shfj['均价（元/平米）']=shfj['总价（万元）']*10000/shfj['建筑面积'] # 注意单位
```


```python
shfj['均价标准化'] = shfj[['均价（元/平米）']].apply(lambda x:(x-x.min())/(x.max() - x.min()),axis=0) # 0~1 标准化
```

### 1.2.3 【物业类型】 列的处理
* 创建虚拟变量


```python
shfj['物业类型'].value_counts()
```




    住宅    228
    别墅     12
    Name: 物业类型, dtype: int64




```python
df = pd.get_dummies(shfj['物业类型']) # 创建虚拟变量 ，用于绘制 热力图
shfj = shfj.join(df) # 按索引连接 
```

### 1.2.4 【户型】【建筑年代】 列处理


```python
shfj['建筑年代']=shfj['年代'].str.split('年',expand=True).get(0).astype(float)
```


```python
(shfj['户型'].str.split('室').map(len)!=2).sum()# 寻找拆分规律
```




    0




```python
shfj['几室'] = shfj['户型'].str.split('室',expand=True)[0].astype(int)
```


```python
shfj['几厅']=shfj['户型'].str.split('室',expand=True)[1].str.split('厅',expand=True)[0].astype(int)
```

### 1.3离散化
* 连续数据离散化，作为查看数据分布使用


```python
shfj['总价分箱'] = pd.cut(shfj['总价（万元）'],bins=[0,50,100,500,1000,12345])
```


```python
shfj['总价分位'] = pd.qcut(shfj['总价（万元）'],q=10,labels=list('abcdefghij'))
```

## 1.4 行列取舍——整理完毕


```python
demos = shfj[['建筑年代','物业费（元/平米）','物业类型','总价（万元）',
              '均价（元/平米）','建筑面积','结构','住宅类别','产权性质','建筑类别','几室','几厅']]
```

# 2 数据分析

## 2.1 数据特征分析


* 相关性分析
* 集中趋势，离散趋势

### 2.1.1集中趋势、离散趋势


```python
demos['总价（万元）'].describe()
```




    count      240.000000
    mean       642.626819
    std       1292.586408
    min         30.600000
    25%        219.998355
    50%        400.000000
    75%        564.500000
    max      11000.000000
    Name: 总价（万元）, dtype: float64




```python
demos['总价（万元）'].plot.box()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x8cbbf90>




![png](output_34_1.png)



```python
demos['总价（万元）'].kurt() # 峰度
```




    40.52782565750391




```python
demos['总价（万元）'].skew() # 偏度
```




    6.0171911532748075



### 2.1.2 相关性分析


```python
# pd.scatter_matrix(demos,figsize=(8,8),marker='+')
plt.scatter(demos['均价（元/平米）'],demos['物业费（元/平米）'])
```




    <matplotlib.collections.PathCollection at 0x8c0a9d0>




![png](output_38_1.png)



```python
u = demos['均价（元/平米）'].mean()
s = demos['均价（元/平米）'].std()
stats.kstest(demos['均价（元/平米）'],'norm',(u,s))#正态性检验
```




    KstestResult(statistic=0.10648411593817408, pvalue=0.007990960793191837)




```python
u1 = demos['物业费（元/平米）'].mean()
s1 = demos['物业费（元/平米）'].std()
stats.kstest(demos['物业费（元/平米）'],'norm',(u,s))#正态性检验
```




    KstestResult(statistic=0.9515180129660263, pvalue=0.0)




```python
demos[['均价（元/平米）','物业费（元/平米）']].corr(method='spearman')# 斯皮尔曼相关系数
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
      <th>均价（元/平米）</th>
      <th>物业费（元/平米）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>均价（元/平米）</th>
      <td>1.000000</td>
      <td>0.096254</td>
    </tr>
    <tr>
      <th>物业费（元/平米）</th>
      <td>0.096254</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



* 物业费与总价，斯皮尔曼相关系数低于0.3，不存在线性相关

## 2.2 可视化分析


* 三维度的散点图：

    

* 箱线图——分布分析


```python
fig = plt.figure(figsize=(12,12))
ax1=fig.add_subplot(2,1,1)
shfj.plot.scatter(y='建筑面积',x='建筑年代',ax=ax1,c=shfj['均价标准化'],cmap='Blues',s=shfj['均价（元/平米）']/100,alpha=0.5)

ax2 = fig.add_subplot(2,1,2)
shfj.dropna().boxplot(column='建筑面积',by='总价分箱',ax=ax2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x987c090>




![png](output_44_1.png)


* 各个装修的房屋成交总价分布
* 低于50平米的房屋成家量很少


```python
sns.stripplot(x='装修',y='总价（万元）',data=shfj,jitter=True,hue='住宅类别',size=8)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x9b94bb0>




![png](output_46_1.png)



```python
zj = shfj['总价分箱'].value_counts(normalize=True)
```


```python
zj.plot.pie(figsize=(6,6),startangle=90,explode=(0,0.5,0.3,0,0))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x9f8bc90>




![png](output_48_1.png)


# 设置表格样式
* 设置极值 背景高亮
* 缺失值填充为指定颜色
* 数字分类设置颜色


```python
def colors(val):
    if val>50000:
        color='red'
    else:
        color='blue'
    return('color:%s'%color)

def colorss(val):
    if val>500:
        color='red'
    elif val<100:
        color = 'blue'        
    else:
        color='pink'
    return('color:%s'%color)

```


```python
demos.head(11).style.\
    applymap(colors,subset=['均价（元/平米）']).\
    applymap(colorss,subset=['总价（万元）']).\
    highlight_max(color='#861000').\
    highlight_min(color='pink').\
    bar(color='#d65f5f',subset=['物业费（元/平米）']).\
    highlight_null(null_color='blue')
```




<style  type="text/css" >
    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col1 {
            : ;
            : ;
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 28.2%, transparent 0%);
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col3 {
            color: blue;
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col4 {
            color: blue;
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col1 {
            : ;
            : ;
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 15.4%, transparent 0%);
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col3 {
            color: pink;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col4 {
            color: red;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col11 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col1 {
            : ;
            : ;
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 79.5%, transparent 0%);
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col3 {
            color: red;
            background-color:  #861000;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col4 {
            color: red;
            background-color:  #861000;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col5 {
            background-color:  #861000;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col6 {
            background-color:  blue;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col9 {
            background-color:  blue;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col10 {
            background-color:  #861000;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col1 {
            : ;
            background-color:  pink;
            width:  10em;
             height:  80%;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col3 {
            color: pink;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col4 {
            color: blue;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col7 {
            background-color:  blue;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col11 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col1 {
            : ;
            background-color:  pink;
            width:  10em;
             height:  80%;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col3 {
            color: pink;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col4 {
            color: blue;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col11 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col0 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col1 {
            : ;
            : ;
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 11.5%, transparent 0%);
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col3 {
            color: pink;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col4 {
            color: red;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col5 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col7 {
            background-color:  blue;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col10 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col0 {
            : ;
            : ;
            background-color:  blue;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col1 {
            : ;
            : ;
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 87.2%, transparent 0%);
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col3 {
            color: red;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col4 {
            color: blue;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col6 {
            background-color:  blue;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col7 {
            background-color:  blue;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col8 {
            background-color:  blue;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col9 {
            background-color:  blue;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col10 {
            background-color:  #861000;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col11 {
            background-color:  #861000;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col1 {
            : ;
            : ;
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 28.2%, transparent 0%);
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col3 {
            color: pink;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col4 {
            color: blue;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col10 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col11 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col1 {
            : ;
            : ;
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 10.3%, transparent 0%);
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col3 {
            color: pink;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col4 {
            color: blue;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col11 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col1 {
            : ;
            : ;
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 2.6%, transparent 0%);
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col3 {
            color: pink;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col4 {
            color: red;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col11 {
            : ;
            background-color:  pink;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col0 {
            background-color:  #861000;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col1 {
            background-color:  #861000;
            : ;
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 100.0%, transparent 0%);
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col3 {
            color: red;
            : ;
            : ;
            : ;
        }    #T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col4 {
            color: red;
            : ;
            : ;
            : ;
        }</style>  
<table id="T_95bfc478_4090_11e9_aa0b_f46d046610b5" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >建筑年代</th> 
        <th class="col_heading level0 col1" >物业费（元/平米）</th> 
        <th class="col_heading level0 col2" >物业类型</th> 
        <th class="col_heading level0 col3" >总价（万元）</th> 
        <th class="col_heading level0 col4" >均价（元/平米）</th> 
        <th class="col_heading level0 col5" >建筑面积</th> 
        <th class="col_heading level0 col6" >结构</th> 
        <th class="col_heading level0 col7" >住宅类别</th> 
        <th class="col_heading level0 col8" >产权性质</th> 
        <th class="col_heading level0 col9" >建筑类别</th> 
        <th class="col_heading level0 col10" >几室</th> 
        <th class="col_heading level0 col11" >几厅</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row0" class="row_heading level0 row0" >0</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col0" class="data row0 col0" >1989</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col1" class="data row0 col1" >1.5</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col2" class="data row0 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col3" class="data row0 col3" >30.6</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col4" class="data row0 col4" >4346.59</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col5" class="data row0 col5" >70.4</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col6" class="data row0 col6" >平层</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col7" class="data row0 col7" >普通住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col8" class="data row0 col8" >个人产权</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col9" class="data row0 col9" >板楼</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col10" class="data row0 col10" >2</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row0_col11" class="data row0 col11" >2</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row1" class="row_heading level0 row1" >1</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col0" class="data row1 col0" >1989</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col1" class="data row1 col1" >1</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col2" class="data row1 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col3" class="data row1 col3" >418</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col4" class="data row1 col4" >83902</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col5" class="data row1 col5" >49.82</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col6" class="data row1 col6" >平层</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col7" class="data row1 col7" >普通住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col8" class="data row1 col8" >个人产权</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col9" class="data row1 col9" >塔楼</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col10" class="data row1 col10" >2</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row1_col11" class="data row1 col11" >1</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row2" class="row_heading level0 row2" >2</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col0" class="data row2 col0" >2005</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col1" class="data row2 col1" >3.5</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col2" class="data row2 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col3" class="data row2 col3" >2650</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col4" class="data row2 col4" >91918.1</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col5" class="data row2 col5" >288.3</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col6" class="data row2 col6" >nan</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col7" class="data row2 col7" >普通住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col8" class="data row2 col8" >个人产权</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col9" class="data row2 col9" >nan</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col10" class="data row2 col10" >4</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row2_col11" class="data row2 col11" >2</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row3" class="row_heading level0 row3" >3</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col0" class="data row3 col0" >1994</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col1" class="data row3 col1" >0.4</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col2" class="data row3 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col3" class="data row3 col3" >280</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col4" class="data row3 col4" >50000</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col5" class="data row3 col5" >56</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col6" class="data row3 col6" >平层</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col7" class="data row3 col7" >nan</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col8" class="data row3 col8" >商品房</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col9" class="data row3 col9" >板楼</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col10" class="data row3 col10" >2</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row3_col11" class="data row3 col11" >1</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row4" class="row_heading level0 row4" >4</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col0" class="data row4 col0" >1998</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col1" class="data row4 col1" >0.4</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col2" class="data row4 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col3" class="data row4 col3" >190</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col4" class="data row4 col4" >32758.6</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col5" class="data row4 col5" >58</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col6" class="data row4 col6" >平层</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col7" class="data row4 col7" >普通住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col8" class="data row4 col8" >个人产权</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col9" class="data row4 col9" >钢混</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col10" class="data row4 col10" >2</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row4_col11" class="data row4 col11" >1</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row5" class="row_heading level0 row5" >5</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col0" class="data row5 col0" >1986</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col1" class="data row5 col1" >0.85</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col2" class="data row5 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col3" class="data row5 col3" >205</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col4" class="data row5 col4" >58571.4</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col5" class="data row5 col5" >35</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col6" class="data row5 col6" >平层</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col7" class="data row5 col7" >nan</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col8" class="data row5 col8" >商品房</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col9" class="data row5 col9" >板楼</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col10" class="data row5 col10" >1</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row5_col11" class="data row5 col11" >2</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row6" class="row_heading level0 row6" >6</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col0" class="data row6 col0" >nan</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col1" class="data row6 col1" >3.8</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col2" class="data row6 col2" >别墅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col3" class="data row6 col3" >1000</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col4" class="data row6 col4" >42194.1</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col5" class="data row6 col5" >237</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col6" class="data row6 col6" >nan</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col7" class="data row6 col7" >nan</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col8" class="data row6 col8" >nan</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col9" class="data row6 col9" >nan</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col10" class="data row6 col10" >4</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row6_col11" class="data row6 col11" >3</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row7" class="row_heading level0 row7" >7</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col0" class="data row7 col0" >1996</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col1" class="data row7 col1" >1.5</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col2" class="data row7 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col3" class="data row7 col3" >230</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col4" class="data row7 col4" >46000</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col5" class="data row7 col5" >50</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col6" class="data row7 col6" >平层</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col7" class="data row7 col7" >普通住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col8" class="data row7 col8" >个人产权</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col9" class="data row7 col9" >板楼</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col10" class="data row7 col10" >1</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row7_col11" class="data row7 col11" >1</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row8" class="row_heading level0 row8" >8</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col0" class="data row8 col0" >1994</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col1" class="data row8 col1" >0.8</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col2" class="data row8 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col3" class="data row8 col3" >273</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col4" class="data row8 col4" >39674.5</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col5" class="data row8 col5" >68.81</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col6" class="data row8 col6" >平层</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col7" class="data row8 col7" >普通住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col8" class="data row8 col8" >商品房</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col9" class="data row8 col9" >塔楼</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col10" class="data row8 col10" >2</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row8_col11" class="data row8 col11" >1</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row9" class="row_heading level0 row9" >9</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col0" class="data row9 col0" >1996</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col1" class="data row9 col1" >0.5</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col2" class="data row9 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col3" class="data row9 col3" >410</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col4" class="data row9 col4" >56944.4</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col5" class="data row9 col5" >72</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col6" class="data row9 col6" >平层</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col7" class="data row9 col7" >普通住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col8" class="data row9 col8" >个人产权</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col9" class="data row9 col9" >板楼</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col10" class="data row9 col10" >2</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row9_col11" class="data row9 col11" >1</td> 
    </tr>    <tr> 
        <th id="T_95bfc478_4090_11e9_aa0b_f46d046610b5level0_row10" class="row_heading level0 row10" >10</th> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col0" class="data row10 col0" >2011</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col1" class="data row10 col1" >4.3</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col2" class="data row10 col2" >住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col3" class="data row10 col3" >1000</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col4" class="data row10 col4" >90909.1</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col5" class="data row10 col5" >110</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col6" class="data row10 col6" >平层</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col7" class="data row10 col7" >普通住宅</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col8" class="data row10 col8" >个人产权</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col9" class="data row10 col9" >板楼</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col10" class="data row10 col10" >2</td> 
        <td id="T_95bfc478_4090_11e9_aa0b_f46d046610b5row10_col11" class="data row10 col11" >2</td> 
    </tr></tbody> 
</table> 




```python
demos.to_csv('上海房屋2017成交信息.csv',encoding='gbk',index=False)
```
