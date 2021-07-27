# [筆記] OXnet: Deep Omni-supervised Thoracic Disease Detection from Chest X-rays

[![hackmd-github-sync-badge](https://hackmd.io/h7qG-v_dRceW6mCQwRjk5A/badge)](https://hackmd.io/h7qG-v_dRceW6mCQwRjk5A)


- MICCAI 2021
- [arxiv](https://arxiv.org/abs/2104.03218)
- [Github](https://github.com/LLYXC/OXnet) (專案目前只有 readme....)


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


### Dual Attention Alignment

{%youtube eJLr__iun20 %}







