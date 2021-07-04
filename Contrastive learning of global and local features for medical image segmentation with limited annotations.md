# Contrastive learning of global and local features for medical image segmentation with limited annotations

- NIPS 2020
- [paper](https://arxiv.org/pdf/2006.10511)


# Methods

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
    - 呼應前面作者提到，不同 volume 的相同 partition 大部分抓到都是同一個器官影像

- $G^{D}$
    - 對於不同 volume 但是來自相同 partion 的 image 應該要有相似的 representation，因為抓到的器官差不多
    - partition-wise representation clusters
    - ![](https://i.imgur.com/FPgWLdR.png =400x)
    - 來自同一個 partition 的距離要越近，來自不同 partition 的距離要越遠
    - negative set 的設計則跟 $G^{D-}$ 一樣

## Local Contrastive Loss



