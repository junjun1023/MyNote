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
- 為了能更好做 segmentation，本篇論文提出 ***Local Contrastive Loss*** 來 pretrain decoder
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

於是作者採取兩種手段，其一是 random sampling，如下圖，有深色框的就是被 sample 到的 image

![](https://i.imgur.com/6MExabm.png =400x)


但如果希望每個 volume 都能被均勻的 sample 出特定張數，該怎麼做？

很簡單，每個 volume 都切成 $S$ 份，每份 sample 一張

![](https://i.imgur.com/mbcjiDl.png =400x)

到目前為止，就成功從 3D 影像取得 2D image input data 了


## Global Contrastive Loss

- 一般熟知的 contrastive loss 是
    - ![](https://i.imgur.com/RSogV2f.png)
    - 對於同一個 image $x$ 做兩種不同 augmentations 得到 $\hat{x}$ 和 $\tilde{x}$
    - 同一個 $x$ 的 embedding 距離越近，不同 $x$ 的 embedding 距離越遠
    - 作者稱呼這種，**隨機**抓兩個不同的 image 計算 positive 和 negative 為 **Random Strategy**
- 作者結合 3D 影像（如：CT, MRI）重新定義了 positive set 和 negative set
    - 結合 domain and problem specific knowledge 可能可以更好的幫助 CL 
    - 利用 domain-specific knowledge 提供 ==global cue== 來學習 volumetric medical images
    - 利用 problem-specific knowledge 提供 ==local cue== 來學習 segmentation
- 假設有 $M$ 個 volumes，每個 volume 由 $Q$ 張 images 組成，把每個 volume 切成 $S$ 個 partitions，每個 partitions 有 $D$ 張 images
    - 一個 volume 可以理解成一組 3D 影像，一組 CT 影像
    - 第 $i$ 個 volume 的==第 $s$ 個 partition== 寫作 $x_{s}^i$
    - 不同 volumes 的同一個 partition 可以視為抓到同一個 anatomical area，每個 volume 都是某個病人的一組 CT 影像，簡單來說就是每個人的器官都長得差不多
    - 所以在 training CL 時，$x_{s}^i = x_{s}^j$

Random Strategy $G^{R}$
: 隨機從所有 volumes 中 sample 出 $N$ 張 images 形成一個 mini-batch

Proposed Strategy $G$
: 基於 $G^R$，計算不同 volume 的 partition 的 similarity
: 先 sample 出 $m$ 個 volumes，$m<M$，對於這 $m$ 個 sample 出的 volume，從每一個 partition 中 sample 出一張 image
: 也就是會有 $m\times S$ 張（$m$ 個 volumes, $S$ 個 partition）

- $G^{D-}$
    - 對於 $x_{s}^i$，第 $i$ 個 volume 的第 $s$ 個 partition 的某張 sample 到的 image，過完 augmentation 會有 pair $(\hat{x_{s}^i}, \tilde{x_{s}^i})$
    - 對於 pair $(\hat{x_{s}^i}, \tilde{x_{s}^i})$，它的 negative set 是![](https://i.imgur.com/FwjeLtb.png =250x) 
    - 意思是，所有 volume 除了第 $s$ 個 partition，其餘都是 negative
    - 而 positive set，$(x_{s}^i, \hat{x_{s}^i}, \tilde{x_{s}^i})$ 兩兩抓都是 positive
    - 呼應前面作者提到，不同 volume 的相同 partition 大部分抓到都是同一個器官影像

- $G^{D}$
    - 對於來自不同 volume 但是相同 partition 的 image 應該要有相似的 representations，因為抓到的器官差不多
    - partition-wise representation clusters
    - ![](https://i.imgur.com/FPgWLdR.png =400x)
    - 來自同一個 partition 的距離要越近，來自不同 partition 的距離要越遠
    - positive set $(x_{s}^i, \hat{x_{s}^i}, \tilde{x_{s}^i})+(x_{s}^i, x_{s}^j)$ （這邊 $x_{s}^j$ augmentation 過後的不能算是 positve，我猜測是因為這邊做的 augmentation 有 crop 之類的）
    - negative set 的設計則跟 $G^{D-}$ 一樣

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