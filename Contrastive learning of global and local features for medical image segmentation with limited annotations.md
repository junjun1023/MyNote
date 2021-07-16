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

$x$：image

$(\hat{x}, \tilde{x})$：image 過兩種不同的 transformation

$e(\hat{x})$：transformed image 過 encoder

$g$：projection head

$g(e(\hat{x}))$：embedding 過 projection head，map 到 normalized space 得到 $\hat{z}$

$e^{sim(\hat{z}, \tilde{z})}$：$\hat{z}, \tilde{z}$ 的相似度取 exponential
:::


這邊補充一個作者提到的觀點
> Corresponding partitions in different volumes can be considered to capture similar anatomical areas

簡單來說，不同的 volume 的同個 partition 抓到的東西都是相似的，強制讓相似的 image embedding 不相似是不合理的，如果不加思索直接套用 contrastive loss，可能會影響 model 的 performance，所以作者重新定義 *positive set* 和 *negative set*


![](https://i.imgur.com/TIHPchH.png)


- $G^{D-}$
    - positive：same image source
    - negative：不同 partition 的 image
    - ![](https://i.imgur.com/8bSNetV.png =200x) （紅色虛線表示 negative）

- $G^{D}$
    - positive
        - same image source
        - 來自同個 partition 的 image
    - negative：不同 partition 的 image
    - ![](https://i.imgur.com/i72HNL7.png =200x) （綠色實線表示 positive，紅色虛線表示 negative）
 
下圖是論文中的對 global contrastive learning $G^{D}$ 的示意圖，annotation 前面講過了就不贅述
![](https://i.imgur.com/kbZDDH1.png)



## Local Contrastive Loss

Global contrastive loss 提供 image-level 的資訊，但是 segmentation 這類 pixel-level 的 prediction 會更需要 local representation 來區別 neighborhoods

- 一個 encoder-decoder 的架構，用 global contrastive loss 來限制 encoder，用 local contrastive loss 來限制 decoder
- 在第 $l$ 個 decoder block 接一個 projection head， 透過 local contrastive loss，使得這些 feature maps 有相似的 local regions 距離比較近，不相似的 local regions 距離比較遠
    - 想當然作者這邊也要定義什麼是相似的 local region？什麼是不相似的 local region？
- 對於一個 input image $x$，過兩種不同 transformations 後得到 $\hat{x}$ 和 $\tilde{x}$，過 encoder 到第 $l$ 個 decoder block，再過 projection head，會分別取得 $\hat{f}=g_{2}(d_{l}(e(\hat{x}))$ 和 $\tilde{f}=g_{2}(d_{l}(e(\tilde{x}))$
    - $d_{l}(e(\hat{x}))$ 出來的 feature maps 維度是 $(W_{1}, W_{2}, C)$
    - 首先把 feature maps 分成 $A$ 個 regions，每個 region 的維度是 $(K, K, C), K<min(W_{1}, W_{2})$
    - 那 $\hat{x}$ 和 $\tilde{x}$ 同個位置的 region 相似，不同位置的 region 不相似
    - ![](https://i.imgur.com/OxkIMzx.png =400x)
    - 廣義來說，定義 global contrastive loss 時有定義 positive set，這邊原本說要是同一個 $x$ 的 pair $(\hat{x}, \tilde{x})$，其實不然，只要是 positive pair 就可以了
    - ![](https://i.imgur.com/NgHrPlq.png)
    - 特別註明這邊用到的 transformation 是只跟 ==intensity== 相關的

Random Strategy $L_{R}$
: 跟 random strategy global contrastive loss 一樣，從所有的 volumes 中 sample 出 $N$ 張 images，過 intensity transformations
: positive pairs $(\hat{f}_{s}^i(u, v), \tilde{f}_{s}^i(u, v))$
: negative pairs **在同個 feature maps $\hat{f}_{s}^i$, $\tilde{f}_{s}^i$ 的其他 regions**
: 前面 global contrastive loss 多考慮不同 volumes 會有相似的 representation，local loss 這邊也可以再算一次不同 volumes 之間的相似 $({f}_{s}^i, {f}_{s}^j)$
: 跟 global contrastive loss 不同的地方在於，local loss 的 postive 包含不同 transformed 版本 $(\hat{f}_{s}^i, \tilde{f}_{s}^j)$ 都可以視為 positive，原因我認為在於 contrastive learning 學習的是不同 intensity 的變化，而 global loss 的 transformation 會包含 random crop, flip 等等，會影響結構
: 另外，雖然原論文沒有特別提到，不過他的 annotations 暗示了 local loss 的 positive 要是在同個 partitions。想想其實滿合理，要在 global 基礎下是相同的找 local 的 positive 跟 negative

## Pretraining

這篇論文的方法雖然可以 end-to-end 的 train，但是作者採用分階段一個一個 train 來避免要一個一個調校超參數

1. 準備一個 encoder-decoder network，例如 UNet
2. Pretrain unet.encoder
    - encoder 接一個 projection head 用來計算 global contrastive loss
    - transformations 包含 crop, flip, intensity, ...
    - train 完後把外加的 projection head 丟掉
3. Pretrain unet.decoder[$\ :l$]
    - freeze unet.encoder
    - unet.decoder 只取到第 $l$ 個 block 並外接一個 projection head
    - 用 local contrastive loss 來 train 這 $l$ 個 blocks
    - transformations 只跟 intensity 有關係
    - train 完後把外加的 projection head 丟掉
4. Finetune network
    - 只用 segmentation loss 來 finetune 整個 network








###### tags: `contrastive learning`, `segmentation`, `volumetric image`