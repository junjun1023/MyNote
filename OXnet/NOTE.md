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

除了 resize，作者還將 local attention 的每個數值都除以 local attention 的總和，好讓整個 local attention 的總和為 $1$，這步驟的目的是為了好讓接下來的 element-wise multiplication 能具有 multiple instance learning 中的 pooling 的效果

![](https://i.imgur.com/ufT8kaw.png)

兩者 element-wise 相乘後的維度是 $(3, 3, 5)$，其實到這個步驟就已經是 dual attention alignment 了，不過為了能 classify，所以透過將 
$3*3$ 個 feature vectors 的加總，得到 $(1, 1, 5)$ 的 feature vector

這條 feature vector 再過 activation function 得到 classification prediction

![](https://i.imgur.com/WRwhokx.png)

到目前為止的整體示意圖如下

![](https://i.imgur.com/QhqTfDm.png)


---

那其實有了 DAA (dual attention alignment) 後，所有在 global 包含的資訊都可以和 local 做結合，所以作者結合 multi-label metric learning 讓 encoder 能學得更好

也就是藉由 prototype learning，讓同個 class 的 feature 更近，不同 class 的 feature 更遠


### Category-specific Feature

{%youtube 2P7sHqD9FCk %}


---



![](https://i.imgur.com/XH5LHFT.png)




![](https://i.imgur.com/9Wz7qGn.png)

![](https://i.imgur.com/SP4B2yi.png)


![](https://i.imgur.com/tFPwI76.png)


![](https://i.imgur.com/T2v5mgU.png)



### Global Prototype Alignment

{%youtube _lKAT_2DDAw %}


![](https://i.imgur.com/WgPTb0G.png)


![](https://i.imgur.com/gIOUo4g.png)

![](https://i.imgur.com/p7pmduA.png)

![](https://i.imgur.com/zaqm7pT.png)
