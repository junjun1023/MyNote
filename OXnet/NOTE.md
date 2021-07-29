# [筆記] OXnet: Deep Omni-supervised Thoracic Disease Detection from Chest X-rays

[![hackmd-github-sync-badge](https://hackmd.io/h7qG-v_dRceW6mCQwRjk5A/badge)](https://hackmd.io/h7qG-v_dRceW6mCQwRjk5A)


- MICCAI 2021
- [arxiv](https://arxiv.org/abs/2104.03218)
- [Github](https://github.com/LLYXC/OXnet) (專案目前只有 readme，作者表示還在整理 code)


---

# Overview

- Supervision granularities (細粒度) 可以依序分為完全沒有 label 的 self-supervised，label 不是很完整 的 semi-supervised，到有 mask 或 bounding box 的 supervised
- 但是到目前為止，似乎沒有一種架構是可以統一不同細粒度的監督來做到胸部疾病檢測
- 這篇論文主要提出框架 OXnet，作者表示「這是第一個 deep omni-supervised (深度全方位監督) 的胸部疾病檢測網路」
- 由於這篇論文的方法比較 specific，沒有辦法一言以蔽之，推薦先有 RetinaNet 模型的概念會更好理解


---

# Mothodology


## RetinaNet

![](https://i.imgur.com/39cbu7V.png)

正式開始進入論文的方法前，先來回憶一下 RetinaNet

使用 FPN 的架構，透過 lateral connection 讓 model 具有更全面的 receptive field，並且在 decoder 不同大小的層上都接出去兩個子網路 (subnet)。

RetinaNet 的任務是物件偵測，事先定義好 anchor，每個 anchor 具有類別 (class) 和偏移量 (offset) 兩個屬性， decoder 接出去的兩個 subnets 各別對應 class 和 offset

### Class Subnet

在 class 這個分支上，模型最後會產生 $(H, W, K*A)$ 維度的 feature map，**K 代表幾個 classes**，**A 代表事先定義好的幾個 anchor**

在原論文中，事先定義好的 anchor 有三種大小 $\{1:2, 1:1, 2:1\}$ 以及三種縮放比例  $\left\{2^{0}\right.$, $\left.2^{1 / 3}, 2^{2 / 3}\right\}$，所以每個類別會有 9 種可能的 anchor，共有 K 個類別，所以才會是 $K*A$

### Box Subnet

因為每個 anchor 都會有偏移的可能，同一個位置 (position，也就是 $H*W$ 的方向看過去) 的偏移量是相同的，所以每個位置都會預測 4 個數值，分別對應四個頂點座標的偏移量 (offset)

## OXnet

因為本篇論文對我來說有點繞，所以我做了影片來表示整個模型設計的架構，觀看影片可以更清楚知道模型的流程，文字敘述的部分我會用影片的截圖來說明

### Global Attention Head

{%youtube IUodqnEHYv8 %}

---

#### Global Attention Head

![](https://i.imgur.com/ypSPRtf.png)


RetinaNet 可以拆成 encoder 和 decoder 兩個部分，而所謂的 global attention head 其實就是直接用 encoder 的輸出 feature 做分類

具體實作也很簡單，就是把 feature 透過 1x1 convolution 降維到 class 數量 (RetinaNet 每個 layer 的 channel 數都是 256)，這邊假設有 5 個 classes

```python=
global_attention = nn.Conv2d(in_channel=256, out_channel=5, kernel_size=1)
```

再透過 global average pooling 把 $(H, W, 5)$ 的 $global\ attention\ \mathcal{X}$ 降低 spatial size 到 $(1, 1, 5)$，也就是對每個 class 的 $(H, W)$ 取 maximum

最後過 activation function 得到模型的 prediction

之所以是 global attention 是因為 encoder 最後一層 feature 的 receptive field 比較全局，所以稱為 global



![](https://i.imgur.com/M02tqdg.png =400x)

(a) ground-truth (b) global attention

這邊直接把 global attention $\mathcal{X}$ 畫出來看，可以發現單純使用 global attention，模型關注的位置非常不合理，又因為 RetinaNet 的 decoder 其實可以得到更 local 的資訊，所以作者也加入 local attention


---


### Dual Attention Alignment

{%youtube 8rIS-z5ag38 %}


---

#### Local Attention
![](https://i.imgur.com/KdStERx.png)


RetinaNet decoder 的 class subnet 會輸出 $(W, H, K * A)$ 維度的 feature maps，每個 class 會有 A 個 feature maps，共有 K 個 class

這步驟是對著每個 class 的 A 個 feature maps 的每個 position 取 max

==p.s. 這篇筆記以及原論文中，所有有下標 $_c$ 的 annotation 都代表某個 class==

![](https://i.imgur.com/cQJsRiD.png =200x)

(a) ground-truth (c) local attention

直接把取完 max 的 local attention 拿來看，可以看到模型確實 focus 在更合理的範圍

不過 global attention 的好處是可以有更全局的 receptive field，作者提出結合 global attention 和 local attention 的 dual attention alignment

接下來的說明都假設一個情境：
1. classes = 5
2. batch size = 4
3. global attention 的 $(W, H)=(3, 3)$
4. local attention 的 $(W, H)=(5, 5)$


#### Dual Attention Alignment

Dual attention alignment 的概念不困難，結合 global attention 和 local attention，所以只要把兩者 element-wise 相乘即可，見下圖


![](https://i.imgur.com/BHilERR.png)

但是因為 local attention 的 spatial size 和 global attention 的 spatial size 不一樣，所以需要先 resize local attention

除了 resize，作者還將每個 class 的 local attention 的每個數值都除以該 class local attention 的總和，讓每個 class 的 local attention 的總和為 $1$，這步驟的目的是為了讓接下來的 element-wise multiplication 能具有 multiple instance learning 中的 pooling 的效果

$\mathcal{R}_{c}=\frac{\mathcal{A}_{c}}{\sum_{i}\left[\mathcal{A}_{c}\right]_{i}}$


![](https://i.imgur.com/ufT8kaw.png)

兩者 element-wise 相乘後的維度是 $(3, 3, 5)$，其實到這個步驟就已經是 dual attention alignment 了，不過為了能 classify，所以透過將 
$3*3$ 個 feature vectors 的加總，得到 $(1, 1, 5)$ 的 feature vector

這條 feature vector 再過 activation function 得到 classification prediction

$p_{c}=\sigma\left(\sum_{i}\left[\mathcal{X}_{c} \odot \mathcal{R}_{c}\right]_{i}\right)$

![](https://i.imgur.com/WRwhokx.png)

到目前為止的整體示意圖如下

![](https://i.imgur.com/QhqTfDm.png)


---

那有了 DAA (dual attention alignment) 後，所有在 global 包含的資訊都可以和 local 做結合，所以作者結合 multi-label metric learning 讓 encoder 能學得更好

藉由 prototype learning，讓同個 class 的 feature 距離較近，不同 class 的 feature 距離較遠，算是一種 cluster 的概念，prototype learning 的目標是找到每個 cluster 的模板 (原型，prototype)

那怎麼決定一個 feature 屬於哪個 cluster 呢 ? 看 feature 跟哪個 prototype 的距離比較近，就屬於那個 prototype 的 cluster

要計算 prototype，首先要知道每個 class 的 feature 落在哪裡，這步驟可以直接處理 global attention $\mathcal{X}$，flatten 每個 class 的 feature map $(H, W)$ 當作 class 的其中一個 feature

不過作者提出一個結合 local attention 的方法來求得 class 的 feature，或稱 category-specific feature


### Category-specific Feature

{%youtube lWr_W1G4jm8 %}


---

因為舉的例子有 5 個 classes，每個 classes 都要找到 prototype，而 $R_c$ (維度 $(5, 3, 3)$) 包含每個 class 的 local features，顯然是可以做結合的好選項

作者將 encoder 過 conv1x1 前的 feature $\mathcal{M}$ 拿來和 $R_c$ 相乘，前者沒有經過降維，保留的資訊相比 global attention $\mathcal{X}$ 更加完整，和 local 的 $R_c$ 結合可以讓 prototype 同時保有兩者的資訊

![](https://i.imgur.com/XH5LHFT.png)

因為每個 class 都要計算 prototype，舉例有 5 個 classes，把 $(5, 3, 3)$ 的 $R_c$ 拆成 5 個 $(3, 3)$ 的 featurs，每個 feature 都分別和 $\mathcal{M}$ 相乘

這樣的意義是，$R_c$ 具有每個 class 的 local 資訊，可以輔助 highlight 出在 $\mathcal{M}$ 上對 class 比較重要的 feature


![](https://i.imgur.com/9Wz7qGn.png)

下面是相乘的示意圖，對著 position 相乘

![](https://i.imgur.com/SP4B2yi.png)

如此一來，就可以成功 highlight 出那些 position 的 features 對於 class 是重要的

![](https://i.imgur.com/6mXtAiD.png)

最後再把每個 class 得到的 weighted features 對著 position 加總，得到 category-specific features $F_c$，而上下這兩個步驟其實就是 weighted sum

不過到這邊我有個疑惑的點，最後對著 position 加總我認為會失去 local 的特徵了，因為「對 class ==不重要==的 feature」的加總結果可能跟「對 class 來說是==重要==的 feature」的加總結果是一樣的

![](https://i.imgur.com/T2v5mgU.png)


總之，透過 global 的 $\mathcal{M}$ 和 local, class-specific 的 $R_c$ 相乘，得到 category-specific features $F_c$

到目前為止，取得 class 的 feature 後，接下來要找 class 的 prototype



### Global Prototype Alignment

{%youtube _lKAT_2DDAw %}

---

而前面講述到的例子是 input 只有一張影像，可以計算該張影像對於每個 class 的 feature，也就是 category-specific features

但是要找 prototype，需要多一點 features 才能找到最能代表該 class 的 prototype，所以只要找到每張影像的 category-specific features $F_c$，就可以找每個 class 的 prototype


![](https://i.imgur.com/WgPTb0G.png)

最理想的情況是可以一口氣找完所有影像的 $F_c$，但是我想可能因為記憶體大小的問題，所以作者局部的在 mini-batch 找 prototype 再隨著 training steps 更新

至於作者怎麼找 prototype ?

輸入一張影像給模型，模型可以透過 global attention 或 dual attention alignment 分類，並輸出這張影像屬於每個類別的機率，這個機率可以視作模型的 confidence

又因為模型對每張影像都會求 5 個 category-specific feature $F_c$，$F_c$ 取得的方法和 $p_c$ 取得的方法又差不多，所以可以視為模型有多大的信心 (confidence) 認為某個 category-specific feature $F_c$ 屬於某個 class

換個角度看，對於一個 class，每張影像都能得到屬於這個 class 的 feature $F_c$，又能得到模型對於 feature $F_c$ 的 confidence，接下來的做法就很值觀了，就是做**加權平均**找 prototype



![](https://i.imgur.com/gIOUo4g.png)


做法如下圖，$F_c$ 乘 $p_c$ 做最後再加起來，也就是加權相加，$\sum_{k}^{K} p_{k} \cdot \mathcal{F}_{k}$

![](https://i.imgur.com/p7pmduA.png)

要計算加權平均，就把加權相加除以權重總合，也就是除以 $\sum_{k}^{K} {p}$，$K$ 是 batch size，以這個 case 來說 K=4

![](https://i.imgur.com/zaqm7pT.png)


前面提及，因為沒有辦法一口氣算完所有影像的 $F_c$，所以計算 mini-batch 再隨著 training step 更新 prototype，原論文的完整公式如下，$\beta$ 的預設值是 0.7

$\mathcal{P}_{t+1}=\beta \cdot \mathcal{P}_{t}+(1-\beta) \cdot \frac{\sum_{k}^{K} p_{k} \cdot \mathcal{F}_{k}}{\sum_{k}^{K} p_{k}}$

我個人很好奇為什麼 $\beta$ 預設是 0.7，不過作者沒有對這個超參數做實驗，不過舉個例子如下 :

$(((0.7P_0 + 0.3P_1)0.7+0.3P_2)0.7+0.3P_3)0.7+0.3P_4 \\
= ((0.49P_0+0.21P_1+0.3P_2)0.7+0.3P_3)0.7+0.3P_4 \\ =
(0.343P_0+0.147P_1+0.21P_2+0.3P_3)0.7+0.3P_4 \\ = 0.2401P_0+0.1029P_1+0.147P_2+0.21P_3+0.3P_4$

雖然一開始 $P_1$ 的影響很大，不過隨著 training step 增加，$P_1$ 的影響力會逐漸被稀釋，變成以當前計算的 prototype $P$ 為主

不過我是好奇 : 一開始模型還不穩定，$P_1$ 在一開始的影響力又相對比較大，這樣不會影響到模型的收斂嗎 ? 這個問題等到介紹完 prototype loss 再來慢慢思考

對於 prototype，前面提到可以想成就是在做 cluster，cluster 講求「類內距離越近越好，類間距離越遠越好」，也就是一個 class 的 cluster 越聚集越好，不同個 class 的 cluster 彼此越疏遠越好


$(6-1)\ \mathcal{L}_{\text {Intra }}=\frac{1}{N} \sum_{c=1}^{N}\left\|\mathcal{F}_{c}-\mathcal{P}_{c}\right\|_{2}^{2}$

首先來看「類內距離越近越好」，也就是對於一個 class 的 cluster 來說，它的每個 feature $F_c$ 都距離它的 prototype 越近越好，簡言之就是計算歐式距離，距離越小，loss 越低

$(6-2)\ \mathcal{L}_{\text {Inter }}=\frac{1}{N(N-1)} \sum_{c=1}^{N} \sum_{0 \leq j \neq c \leq N}^{N} \max \left(0, \delta-\left\|\mathcal{F}_{j}-\mathcal{P}_{c}\right\|_{2}^{2}\right)$

再來看「類間距離越遠越好」，$\delta$ 是超參數，是 inter-class margin，表示類間的距離，公式拆解來看如下

$\sum_{0 \leq j \neq c \leq N}^{N} \max \left(0, \delta-\left\|\mathcal{F}_{j}-\mathcal{P}_{c}\right\|_{2}^{2}\right)$

$\left\|\mathcal{F}_{j}-\mathcal{P}_{c}\right\|_{2}^{2}$ 是==不屬於當前 class 的 feature $F_j$== 和當前 class 的 prototype 的距離，$\left\|\mathcal{F}_{j}-\mathcal{P}_{c}\right\|_{2}^{2}$ 越大，$\delta-\left\|\mathcal{F}_{j}-\mathcal{P}_{c}\right\|_{2}^{2}$ 會越小，直到 $\left\|\mathcal{F}_{j}-\mathcal{P}_{c}\right\|_{2}^{2} > \delta$ 會有 loss 為 0，也就是兩個 clusters 已經達到預期隔開的程度

剩下外部的公式就很好理解了，$\sum_{c=1}^{N}$ 每個 cluster 都要跟其他所有 cluster $c$ 的 feature $F_j$ 計算，$\frac{1}{N(N-1)}$ 總共會計算 $N(N+1)$ 次取平均

![](https://i.imgur.com/hn1GkOL.png)

## Bounding Box

到目前為止所講解的都是跟分類有關，例如 : global attention 和 dual attention alignment，或是講到

## Unsupervised Knowledge Distillation


## Unified Training