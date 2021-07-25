# PROTOTYPICAL CONTRASTIVE LEARNING OF UNSUPERVISED REPRESENTATIONS

[![hackmd-github-sync-badge](https://hackmd.io/MnJmr14xTeupgxnbpeDaTw/badge)](https://hackmd.io/MnJmr14xTeupgxnbpeDaTw)


- ICLR 2021 poster
- [arxiv](https://arxiv.org/abs/2005.04966), [open review](https://openreview.net/forum?id=KmykpuSrjcq)
- [Github](https://github.com/salesforce/PCL)

# Overview

- Instance discrimination tasks 包含 1. image transformation 2. contrastive loss 兩個要素（我看起來就是 contrastive learning）
- 現有 instance discrimination 的方法大都有同一個缺點：缺乏 semantic 的資訊
    - ![](https://i.imgur.com/U5ATGQV.png =200x)
    - 如上圖，類別 ground-truth label 是馬，但是其實可以再細分為「包含人的馬」跟「不包含人的馬」，而「包含人的馬」的所有影像的 embedding 距離應該要相對更近一些（相似度相對更高）；但如果是一般的 instance discrimination，只在意類別為馬的所有影像 embedding 距離是否都足夠靠近
    - [Supervised Contrastive Learning]() 就透過 label 的 samantic 來限制 embedding 距離遠近，不過是 supervised，反倒浪費了 CL 不需要 label 的特性
- 本篇架構有 encoder 和 momentum encoder
    - 透過 momentum encoder 取得 embeddings
    - 用 k-means 分群（clustering），分別計算各 cluster 的 centroid 後
    - 透過 $ProtoNCE$ 約束 encoder 取得的 embeddings
    - 更新 momentum encoder

# Methodology

![](https://i.imgur.com/8Pocnbe.png)

1. 透過 momentum encoder 取得 embeddings
2. 用 k-means 分群（clustering），分別計算各 cluster 的 centroid 後
3. 透過 $ProtoNCE$ 約束 encoder 取得的 embeddings
4. 更新 momentum encoder



本篇論文方法透過 EM algorithm 最佳化，E step 找到各 cluster 的 centroid (2)，M step 藉由 $ProtoNCE$ 訓練更新 encoder (3)，最後再更新 momentum encoder

---

一般 instance discrimination 使用的 $InfoNCE$ 是計算同張 image 和 不同張 image 的 contrastive

$\mathcal{L}_{\text {InfoNCE }}=\sum_{i=1}^{n}-\log \frac{\exp \left(v_{i} \cdot v_{i}^{\prime} / \tau\right)}{\sum_{j=0}^{r} \exp \left(v_{i} \cdot v_{j}^{\prime} / \tau\right)}$

:::success
Annotations
- 下標 $_i$, $_j$ 表示兩張不同的 images $i$, $j$
- $v_i$, $v_i'$ : 某張 image $i$ 做不同 transformations 的 embedding (vector)
:::

## Prototype Contrastive

本篇論文提出的 $ProtoNCE$ 不僅對比 embedding，還要對比 prototype $c$，見下方式子

$\theta^{*}=\underset{\theta}{\arg \min } \sum_{i=1}^{n}-\log \frac{\exp \left(v_{i} \cdot c_{s} / \phi_{s}\right)}{\sum_{j=1}^{k} \exp \left(v_{i} \cdot c_{j} / \phi_{j}\right)}$

:::success
Annotations
- $v_i$ : 某張 image $i$ 的 embedding
- $c_s$ : 某個 cluster $s$ 的 prototype
- $\phi_{s}$ : 某個 cluster $s$ 有多集中
:::

透過 k-means 分 $M$ 群後找到該群的 prototype $c$，當 embedding $v_i$ 屬於 prototype $c_i$ 的 cluster，則 embedding $v_i$ 越靠近 prototype $c_i$，當 embedding $v_i$ 不屬於 prototype $c_j$ 的 cluster，則 embedding $v_i$ 越遠離 $c_j$


$\phi=\frac{\sum_{z=1}^{Z}\left\|v_{z}^{\prime}-c\right\|_{2}}{Z \log (Z+\alpha)}$

:::success
Annotations
- $z$ : 某個 cluster 的 feature points，一個 cluster 共有 $Z$ 個 feature points
:::

透過 $\phi$ 描述一個 cluster 有多集中，$\phi$ 越小越集中，越大越不集中


## $\mathcal{L}_{\text {ProtoNCE }}$

這個概念其實不太需要數學推導，還算好想像，下面來看完整的 $\mathcal{L}_{\text {ProtoNCE }}$

$\mathcal{L}_{\text {ProtoNCE }}=\sum_{i=1}^{n}-\left(\log \frac{\exp \left(v_{i} \cdot v_{i}^{\prime} / \tau\right)}{\sum_{j=0}^{r} \exp \left(v_{i} \cdot v_{j}^{\prime} / \tau\right)}+\frac{1}{M} \sum_{m=1}^{M} \log \frac{\exp \left(v_{i} \cdot c_{s}^{m} / \phi_{s}^{m}\right)}{\sum_{j=0}^{r} \exp \left(v_{i} \cdot c_{j}^{m} / \phi_{j}^{m}\right)}\right)$

整個 $\mathcal{L}_{\text {ProtoNCE }}$ 由兩部分組成，前半是熟悉的 $\mathcal{L}_{\text {InfoNCE }}$，後半則是計算 cluster contrastive，之所以需要計算 $\frac{1}{M} \sum_{m=1}^{M}$ 的 cluster contrastive 是因為總共分群 $M$ 次，每次分成不同數量個 clusters，目的是為了 encode 出 heirarchical structure。
> We also cluster the samples M times with different number of clusters $K=\left\{k_{m}\right\}_{m=1}^{M}$, which enjoys a more robust probability estimation of prototypes that encode the hierarchical structure. 

其實就是一個大的 cluster 裡面可以再區分出好幾個 sub-cluster，例如 : 類別為馬的影像，其實還可以再區分出「包含人的馬」和「不包含人的馬」



## 數學推導


- 待補



# Results

Appendix Figure 6.
: 綠色框的圖片是從 $K=100k$ 程度的 cluster 取的，橘色框的圖片是從 $K=50k$ 程度的 cluster 取的，$K$ 代表有 $K$ 個 clusters

![](https://i.imgur.com/kUbrnvV.png =400x)



---



###### tags: `contrastive learning`, `prototypical network`






