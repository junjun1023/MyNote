# 數學

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
## CNN Triplet Loss
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