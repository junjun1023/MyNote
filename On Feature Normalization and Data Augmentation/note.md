# On Feature Normalization and Data Augmentation

[![hackmd-github-sync-badge](https://hackmd.io/fmX3tLnzS4Sdiq5SaEAzYQ/badge)](https://hackmd.io/fmX3tLnzS4Sdiq5SaEAzYQ)

- CVPR 2021
- [arxiv](https://arxiv.org/abs/2002.11102)
- [Github](https://github.com/Boyiliee/MoEx)

---

# Overview

- Normalization
    - batch_norm 跟 instance_norm 其實就是對不同維度做 normalization，batch_norm 對整個 batch 做 normalization，instance_norm 對 channel 做 normalization
    - Batch normalization 常常被用在 training 時做 feature scaling
        - 一個 batch 的 features 減 $\mu$ 除以 $\sigma$ 做標準化 (平移縮放)，$\tilde{z}=\frac{Z-\mu}{\sigma}$
        - 再乘 $\gamma$ 加上 $\beta$ 來調整 features 不要都落在 $[0, 1]$，$\hat{z}^{i}=\gamma \odot \tilde{z}^{i}+\beta$
    - Instance normalization 反而在 image generation 這類的 task 表現的比 batch_norm 更好，因為透過 instance_norm 或是 position_norm 的 moments ($\mu$, $\sigma$) 可以更好的抓到 style 和 shape
        - 我看過一個說法是，在 image generation task 更看重 instance 之間的**差異**，要生成一張 image 不需要參考其他 image 的資訊，**應該保有該張 image 的特色，也就是 $\mu$, $\sigma$**
    - ![](https://i.imgur.com/J1yOapo.png)
- 在 image recognition (classification) task 中，latent feature 的 $\mu$, $\sigma$ 被視為 noise，需要透過 batch_norm 移除；但是 image generation task，latent feature 的 $\mu$, $\sigma$ 是一種 feature
    - 例如，下圖 1. 透過 postion normalization 得到 ResNet-18 第一個 layer 的 $\mu$, $\sigma$，仍舊可以透過 $\mu$, $\sigma$ 來預測 class 的類別
        1. ![](https://i.imgur.com/L31Cmop.png =250x)
        2. ![](https://i.imgur.com/EPLDNxs.gif)
    - 比較下表中 classification task 的 error rate，單純從 moments 來做分類 (PONO moments, 紅色) 已經比隨機亂猜 (Random Baseline, 灰色) 來得更好。如果把 moments 移掉 (PONO normalized, 藍色)，結果會比標準的 PONO (綠色) 還要更爛，所以 moments 其實是重要的 feature
        - ![](https://i.imgur.com/qUFjuRx.png =400x)
- 本篇論文的方法基於 positional normalization，既然 moments 代表 shape 和 style，那只要交換 moments 就能限制模型同時學習 a instance 的 feature dist 和 b instance 的 moments
    - 這篇的方法稱為 Moment Exchange (MoEX)
    - ![](https://i.imgur.com/G4NgVJ5.png)

# Methodology

...前面其實已經把本篇論文的核心給講完了，接著細探這篇的方法。但不得不說，這篇的作法跟 [DOMAIN GENERALIZATION WITH MIXSTYLE](https://arxiv.org/abs/2104.02008) 超級無敵像，只差在這篇透過 **intra-instance normalization** 得到 $\mu$, $\sigma$，[DOMAIN GENERALIZATION WITH MIXSTYLE](https://arxiv.org/abs/2104.02008) 是透過 instance normalization 得到 $\mu$, $\sigma$

## MoEX

![](https://i.imgur.com/G4NgVJ5.png)

input $X_A$ 經過模型得到 features $h_A$，透過 insta-instance normalization 取得 $h_A$ 的 $(\mu_A, \sigma_A)$；同時，input $X_B$ 的 features $h_B$ 也會透過 insta-instance normalization 取得 $(\mu_B, \sigma_B)$

- Normalize $A$ 的 features，再透過 $(\mu_B, \sigma_B)$ 縮放平移
- Normalize $B$ 的 features，再透過 $(\mu_A, \sigma_A)$ 縮放平移

透過 normalized $h_A$ 替換成 $(\mu_B, \sigma_B)$ 可以強迫模型同時關注 data 的兩個面向：normalized features 和 moments

## Ground-truth

$y = \lambda \cdot \ell\left(\mathbf{h}_{A}^{(B)}, y_{A}\right)+(1-\lambda) \cdot \ell\left(\mathbf{h}_{A}^{(B)}, y_{B}\right)$, $\lambda \in[0,1]$

既然結合兩個 instances 的特徵，那結合後的 ground-truth 一定是兩個 instances 個別 ground-truth 的 weighted sum

原 paper 的 Table 10. 有做 $\lambda$ 的參數實驗，總的來說最推薦 $\lambda=0.9$


## Normalization

這篇方法有個重要的前提：normalize 只能做在 instance 內部，也就是 intra-instance normalization，例如 positional normalization 就是拿 instance 的 position 做標準化

$\left(\hat{\mathbf{h}}_{i}^{\ell}, \boldsymbol{\mu}_{i}^{\ell}, \boldsymbol{\sigma}_{i}^{\ell}\right)=F\left(\mathbf{h}_{i}^{\ell}\right)$

:::success
Annotation
- $_i$ : 第 $i$ 個 input $x_i$
- $l$ : 第 $l$ 個 layer 的 feature $\mathbf{h}_{i}^{\ell}$
:::



- 假設有個 function $F$ 負責做 intra-instance normalization，給 $F$ feature maps 會得到 
(1) normalized feature $\hat{\mathbf{h}}_{i}^{\ell}$,
(2) $\boldsymbol{\mu}_{i}^{\ell}$,
(3) $\boldsymbol{\sigma}_{i}^{\ell}$
    - 以上方圖示意舉例，$\left(\hat{\mathbf{h}}_{A}, \boldsymbol{\mu}_{A}, \boldsymbol{\sigma}_{A}^{}\right)=F\left(\mathbf{h}_{A}\right)$

$\mathbf{h}_{i}^{\ell}=F^{-1}\left(\hat{\mathbf{h}}_{i}^{\ell}, \boldsymbol{\mu}_{i}^{\ell}, \boldsymbol{\sigma}_{i}^{\ell}\right)$
:::success
Annotation
- $_i$ : 第 $i$ 個 input $x_i$
- $l$ : 第 $l$ 個 layer 的 feature $\mathbf{h}_{i}^{\ell}$
:::

- 對應 $F^{-1}$ 負責從 normalization 還原，給 $F^{-1}$ $(1) \hat{\mathbf{h}}_{i}^{\ell}, (2) \boldsymbol{\mu}_{i}^{\ell}, (3) \boldsymbol{\sigma}_{i}^{\ell}$，做縮放平移
    - 以上方圖示意舉例，$\mathbf{h}_{A}^{(B)}=F^{-1}\left(\hat{\mathbf{h}}_{A}, \boldsymbol{\mu}_{B}, \boldsymbol{\sigma}_{B}\right)$



MoEX 限制在 instance **內**的標準化，很單純的調整 $\boldsymbol{\mu}_{i}^{\ell}, \boldsymbol{\sigma}_{i}^{\ell}$，所以還是可以做 inter-instance normalization (e.g. batch normalization)，也就是 instance **間** 的標準化

Intra-instance normalization 有很多種 (IN, GN, LN)，作者也有做相關的實驗在 Table 8.


# My Conclusions
- 我真的覺得這篇的作法跟 [DOMAIN GENERALIZATION WITH MIXSTYLE](https://arxiv.org/abs/2104.02008) 超級像，原則上是一模一樣，只是切入的角度有一點不同，投稿在不同的 conference 上