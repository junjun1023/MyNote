# 數學
###### tags: `note`

# 前處理
https://ir.nctu.edu.tw/bitstream/11536/68068/7/251107.pdf
![](https://i.imgur.com/HBjiWqe.jpg)
把一個分佈，用幾個guassian model去近似表示它
把資料分群的概念

例：background subtraction



# 統計圖表
## Boxplot
![](https://i.imgur.com/5wohjJ8.png)


* 直線是指 **資料集的所有資料** 
* 最低的橫線是 **資料集的minimum**
* 最高的橫線是 **資料集的maximum**
* 箱子內的橫線是 **資料集合的50% (即是平均值mean)**
* 箱子上方的橫線是 **資料集合的75%**
* 箱子下的的橫線是 **資料集合的25%**
* ( PS. 25% ≠ 一次標準差 )

# Machine learning
## 餘弦退火
https://zhuanlan.zhihu.com/p/93648558
![](https://i.imgur.com/DwvlPI5.jpg)

## Loss functions
### CNN Triplet Loss
(reference) https://medium.com/@CinnamonAITaiwan/cnn%E6%A8%A1%E5%9E%8B-%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8-loss-function-647e13956c50

> 
> Triplet Loss 是Metric Learning中最常用到的Loss之一，Metric Learning主要用來訓練一個能夠比較相似度的模型，有別於一般的Softmax+CE輸出是類別數，Metric Learning的輸出是一個壓縮的維度，因此可以用來壓縮特徵，這樣的好處是我們的模型不需要每一個類別都有大量的訓練集，舉個例子來說，今天我們要學習一個人臉辨識的模型，需要辨識全球的名人，使用Softmax+CE我們總結一共有8000個名人(假設)，因此訓練一個8000個分類的分類器，結果隔了一個月又多了5個名人，此時我們只能重新訓練一個模型或是透過Transfer Learning ，拔掉原來的FC再加上一個8005類的FC。
> 透過Metric Learning，我們就能解決上述遇到的問題，我們學習了一個將人臉壓縮的模型，假設輸出是256維，我們模型再做的事就是將一張人臉照片壓縮到這256維，並且不同的人臉在空間上的距離較遠。

簡單來說，不是train分類，而是train出一個**feature向量**。
用這個向量就可算出**相似度**
![](https://i.imgur.com/qln7TP4.png)

> Triplet Loss要做的事情，就是盡量將組內的sample拉近，組間的sample距離拉遠，有沒有覺得很眼熟！其實觀念就跟LDA(Linear Discriminant Analysis)非常類似，公式如下，其中要注意的一點是，Triplet Loss 的輸入是經過L2-normalization 的特徵：

train 使相似的feature距離靠近

![](https://i.imgur.com/P0cpeYX.png)
> 
> 而一般我們在用的Triplet Loss為Hard Triplet Loss或是Semi-hard Triplet Loss，Hard Triplet Loss意指選取最遠的組內Sample，白話文說的話就是：『最不像的同類樣品(In Batch)間距離，還是要比不同類的樣品還要近』，這樣的缺點是，當資料不夠乾淨時模型會難以收斂，因此目前普遍是使用Semi-hard Triplet Loss

### 0-1 loss function and hinge loss
![](https://i.imgur.com/lnVqtYG.png)

### smooth 0-1 loss function
把0-1 loss變成可以微分

### large-margin softmax loss
https://blog.csdn.net/u014380165/article/details/76864572

tl;dr 原本的Softmax loss會希望把class分開，
LM softmax loss希望不同的類別可以分更開
![](https://i.imgur.com/TrOVvus.png)


### the adaptive robust loss function
http://proceedings.mlr.press/v28/nguyen13a.pdf

這篇數學很多很可怕

![](https://i.imgur.com/lybEtKF.png)

## Clustering
Normalized Mutual Information
https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf


# 統計學
## 假設檢定(hypothesis testing)
H~1~ 對立假設（alternative hypothesis)  => 我想要證明是對的事情
H~0~ 虛無假設（null hypothesis） => 和我想證明是對的事情「相反」的事情

藉由推翻H~0~，可證明H~1~正確


**Type 1 , Type 2 Error**
![type 1,2 error](https://lh3.googleusercontent.com/proxy/ragyvHp72Rf34yyfPn6P2cAiS-ZkkVPZTbw7vtzexpkSs8_-yMXg1sW7fHsW15pp8jV1Sq_-x7V2yCzVJwQeawS6qdq5N1G8q20alNeNPVjxopHSzv_g3Hh3Hw)

Type 1 Error: 當我的H~0~是對的，我卻推翻了他。對應下圖 $\alpha$，落入 $\alpha$ 的機率小於*p-value*
Type 2 Error: 當我的H~0~是錯的，我卻接受了他。對應下圖 $\beta$
$\alpha$ 和 $\beta$ 會負相關(見下圖)

舉例：
Type 1 error為被告無罪，卻被判刑；type 2 error為有罪，卻「沒有」被判刑。
減少冤獄的機率，可能就會增加逃過法律制裁的機率

![type 1,2 error](https://4.bp.blogspot.com/-SltJw0rbF8c/W_Y0xoKG4QI/AAAAAAAAraE/Cb2i1wH8tjw02nMyH7uP-c0i2iksK149QCLcBGAs/s1600/error.png)

(source) https://lhw-cn.blogspot.com/2018/11/type-i-error-and-type-ii-error.html

## P-value
(source) https://researcher20.com/2009/11/16/p%E5%80%BC%E7%9A%84%E8%BF%B7%E6%80%9D%EF%BC%9A%E9%A1%AF%E8%91%97%E8%88%87%E9%9D%9E%E5%B8%B8%E9%A1%AF%E8%91%97/

## 相對風險 (relative risk)
![](https://i.imgur.com/lL0AG2g.png)
source(wiki)

## 勝算比 (odds ratio)
![image alt](https://pic.pimg.tw/dasanlin888/1361874881-3615390424.jpg)

「勝算」（odds）定義是「兩個機率相除的比值」。

以上圖舉例，假設:
*(p.s. 可以把X疾病想成抽菸，Y疾病想成肺癌，可能比較好想像)*

有X疾病的人(Disease)，暴露在Y疾病(Exposed)的機率是30% ($\frac{A}{N3}$)
沒有X疾病的人(Disease)，暴露在Y疾病(Exposed)的機率是40% ($\frac{B}{N4}$)
對有X疾病人來說，暴露在Y疾病的勝算(odds)是$\frac{0.3}{0.7}$ ($\frac{\frac{A}{N3}}{\frac{C}{N3}}$)
對沒有X疾病的人來說，暴露在Y疾病的勝算(odds)是$\frac{0.4}{0.6}$ ($\frac{\frac{C}{N4}}{\frac{D}{N4}}$)

而「有X疾病的人暴露在Y」和「沒有X疾病的人暴露在Y」的
$\begin{equation}勝算比 = \frac{有X疾病人來說，暴露在Y疾病的勝算}{沒有X疾病人來說，暴露在Y疾病的勝算} = \frac{0.3 * 0.6}{0.7 * 0.4} = 0.857\end{equation}$

**注意**
**勝算比不能當成相對風險來解釋**
有X疾病暴露在Y的風險「不是」沒有X疾病暴露在Y疾病的風險 的 0.857 倍
僅能證明風險較高或風險較低

## 勝算比(OR) vs 相對風險(RR)
![](https://i.imgur.com/ciMX6II.png)
source(wiki)

RR會有倍數的關係，OR沒有倍數的關係

## 風險比(Hazard Ratio)

From Wiki:
**風險比率**反映了**單位時間內的相對風險**。相對風險反映的是整個實驗的累積風險，而風險比率能夠反映**每個時間點上的瞬間風險**
![](https://i.imgur.com/arCSFvZ.png)
(source)https://www.cde.org.tw/Content/Files/Knowledge/b0801107-dd21-41ef-8c7d-4c6c2edf1bd7.pdf

## 常態分佈（Normal Distribution)
![](https://i.imgur.com/tPlTPl8.png)
(source Wiki)

**Standard Normal Distribution**
Mean $\mu$ = 0
Variance $\sigma^2$ = 1

## 中央極限定理 Central limit theorem
當很多個（Ｎ＞３０）機率分布平均起來，經由標準化後，機率分布會收斂於常態分布。
![](https://i.imgur.com/OmEZs6k.png)
(source)https://medium.com/@birajparikh/what-is-central-limit-theorem-clt-db3679433dcb

Why do we need Central limit theorem?
有些檢定需要常態分布方能使用。（例如：p-value...)