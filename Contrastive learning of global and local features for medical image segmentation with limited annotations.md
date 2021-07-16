# Contrastive learning of global and local features for medical image segmentation with limited annotations

[![hackmd-github-sync-badge](https://hackmd.io/UAhbq6IEQQGCWAahro0DLg/badge)](https://hackmd.io/UAhbq6IEQQGCWAahro0DLg)


- NIPS 2020
- [arxiv](https://arxiv.org/pdf/2006.10511), [NIPS 2020](https://proceedings.neurips.cc/paper/2020/file/949686ecef4ee20a62d16b4a2d7ccca3-Paper.pdf)
- [Github](https://github.com/krishnabits001/domain_specific_cl)


# Overview

- 一般熟悉 Contrastive Loss 的概念是「一張 image 做兩種不同的 transformations (augmentation)，得到的兩張 images 的 embeddings，他們的**距離 (相似度) 要越近越好**；不同的 images，embedding 的距離 (相似度) 要越遠越好
    - ![](https://i.imgur.com/dPSWhMO.png =300x)
- 因為是拿「同一張 image 和不同張 images」做 contrastive（對比），model 學習到的是 global (image-level) 的概念
- 而當 down stream task 是 segmentation 之類的 task，model 透過 CL 學習到的 representation (embedding) 顯然無法直接用來做 segmentation
- 所以 CL 可以視作 pretrain model 的一種手段，如上述，而這個 model 其實就是 encoder-decoder 架構中的 encoder
    - ![](https://i.imgur.com/Dl3F2sl.png =400x)
- 此外，本篇論文提出 ***Local Contrastive Loss*** 來 pretrain decoder
- 為了將 segmentation task 實現於將 3D medical image (e.g. MRI, CT)，作者也提出一些 sampling 的策略將 volumetric image 轉為 2D image



# Methodology

本篇方法共有兩種 contrastive loss，分別是 global contrastive loss 和 local contrastive loss，這兩種 loss 可以一起用來 pretrain model，但是為免調參需求，作者採用階段式訓練：

1. pretrain encoder with global contrastive loss
2. pretrain decoder with local contrastive loss
3. fine-tuning model (encoder-decoder) with segmentation loss

因為 contrative 的概念很著重於 *positive set* 和 *negative set*，應用於 volumetric image，作者在不同的 stage 提出不同 *positive set* 和 *negative set* 的策略

為了方便理解，下文敘述也是分階段式，可以對應作者的 stage

## Volumetric to Image

首先，將 3D 影像 (volume) 轉成 2D image (slice)，最為直觀的方法是，直接將 volume 切成很多個 slices，就可以當成 input data 丟給 segmentation model

![](https://i.imgur.com/1WNSbgM.png =300x)



不過論文中，作者提到
> Volumetric images of the same anatomical region for different subjects have similar content

本人才疏學淺理解為「一組 3D 影像的每個 slice 抓到的內容其實都差不多」，簡言之就是**因為同一個 volume 的 image 都很像，所以不打算每個 image 都用**

於是作者採取兩種手段，其一是 random sampling，作者稱這種 sampling strategy 為 ***Random Strategy $G^R$***。如下圖，有深色框的就是被 sample 到的 image

![](https://i.imgur.com/6MExabm.png =400x)


但如果希望每個 volume 都能被均勻的 sample 出特定張數，該怎麼做？

很簡單，每個 volume 都切成 $S$ 份（partition），每份 sample 一張。這個做法很聰明，因為每組 3D 影像的++張數++其實都不一樣，很好地確保了每組 3D 影像中得到的 2D images 數量是一樣的

![](https://i.imgur.com/mbcjiDl.png =400x)

:::success
Annotation

$x_{s}^i$：第 $i$ 個 volume 的第 $s$ 個 partition 的那張被 sample 到的 image
:::

到目前為止，就成功從 3D 影像取得 2D image input data 了


## Global Contrastive Loss

解決 input data 的問題後，正式進入第一階段：pretrain encdoer with glocal contrastive loss

這部分其實不難理解，就是一般我們熟悉的 contrastive loss，讓同個 image source 的 embeddings 距離越近越好，讓不同 image source 的 embeddings 距離越遠越好

- ![](https://i.imgur.com/Dl3F2sl.png =400x)
- ![](https://i.imgur.com/RSogV2f.png)

:::success
Annotations

- $x$：image
- $(\hat{x}, \tilde{x})$：image 過兩種不同的 transformation
- $e(\hat{x})$：transformed image 過 encoder
- $g$：projection head
- $g(e(\hat{x}))$：embedding 過 projection head，map 到 normalized space 得到 $\hat{z}$
- $e^{sim(\hat{z}, \tilde{z})}$：$\hat{z}, \tilde{z}$ 的相似度取 exponential
- 這階段的 augmentations 沒有什麼限制
:::


這邊補充一個作者提到的觀點
> Corresponding partitions in different volumes can be considered to capture similar anatomical areas

簡單來說，不同的 volume 的同個 partition 抓到的東西都是相似的，強制讓相似的 image embedding 不相似是不合理的，如果不加思索直接套用 contrastive loss，可能會影響 model 的 performance，所以作者重新定義 *positive set* 和 *negative set*


![](https://i.imgur.com/TIHPchH.png)


### $G^{D-}$
- positive：same image source
- negative：不同 partition 的 image
- ![](https://i.imgur.com/8bSNetV.png =200x) （紅色虛線表示 negative）

### $G^{D}$
- positive
    - same image source
    - 來自同個 partition 的 image
- negative：不同 partition 的 image
- ![](https://i.imgur.com/i72HNL7.png =200x) （綠色實線表示 positive，紅色虛線表示 negative）

### Global Contrastive Diagram


下圖是論文中的對 global contrastive learning $G^{D}$ 的示意圖，annotation 前面講過了就不贅述
![](https://i.imgur.com/kbZDDH1.png)



## Local Contrastive Loss

Global contrastive loss 提供 image-level 的資訊，但是 segmentation 這類 pixel-level 的 prediction 需要 local 的資訊，所以作者提出 **Local Contrastive Loss** 來 pretrain decoder

這邊稍微提一下，decoder 的架構其實是由多個 decoder blocks 組成，要 pretrain 的不是一整個 decoder，而是前 $l$ 個 decoder blocks；另外，在 pretrain decoder blocks 的時候，要固定 encoder 的參數；這個階段的 augmentations 限制在 intensity augmentations

local contrastive 的概念來自一張 image 內部的 pixel-pixel contrastive，也就是「位置」的對比。同一張 image 做了兩種不同 intensity augmentations，兩張 images 相同位置的 pixel (patch) 是 positive，不同位置的 pixel (patch) 是 negative

![](https://i.imgur.com/FpBRkUT.png =300x)

因為是 pretrain $l$ 個 decoder blocks，所以是用 feature maps 來計算 local contrastive loss，以上方示意圖為例，假設 feature maps 的維度是 $(C, H, W)$，對著 $H, W$ 切後取得 **位置（region）**
的 embedding $(C, K, K)$

因為前述提及，不同 volume 的同個 partition 抓到的內容差不多，所以作者在 local contrastive 也有再定義 *positive set* 和 *negative set*

![](https://i.imgur.com/SszAmWo.png =500x)

### Random Strategy
- positive：同個 image source 的==相同 region==
- negative：同個 image source 的==不同 region==
- ![](https://i.imgur.com/l4MfcDw.png =250x)


### $L^D$
- positive：同個 partition 的相同 region
- negative：同個 partition 的不同 region
- ![](https://i.imgur.com/BFzyR1e.png =250x)


### Local Contrastive Diagram

下圖是論文中的對 local contrastive loss 的示意圖

![](https://i.imgur.com/lSFboqX.png)



# My Conclusions

我個人覺得 local contrastive 的概念很厲害，可能可以往 X-ray 影像有哪些 local 的 feature 去 study


###### tags: `contrastive learning`, `segmentation`, `volumetric image`