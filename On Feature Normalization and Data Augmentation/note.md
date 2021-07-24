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
        - 一個 batch 的 features 減 $\mu$ 除以 $\sigma$ 做標準化 (平移縮放)，$\tilde{z}=\frac{Z-\mu}{\sigma}$。再乘 $\gamma$ 加上 $\beta$ 來調整 features 不要都落在 $[0, 1]$，$\hat{z}^{i}=\gamma \odot \tilde{z}^{i}+\beta$
    - Instance normalization 反而在 image generative 這類的 task 表現的比 batch_norm 更好，因為透過 instance_norm 或是 position_norm 的 moments ($\mu$, $\sigma$) 可以更好的抓到 style 和 shape
        - 我看過一個說法是，在 image generation task 更看重 instance 之間的差異，要生成一張 image 不需要參考其他 image 的資訊，應該保有該張 image 的特色，也就是 $\mu$, $\sigma$
    - ![](https://i.imgur.com/J1yOapo.png)
- 在 image recognition (classification) task 中，latent feature 的 $\mu$, $\sigma$ 被視為 noise，需要透過 batch_norm 移除；但是 image generation task，latent feature 的 $\mu$, $\sigma$ 是一種 feature
    - 例如，下圖 1. 透過 postion normalization 得到 ResNet-18 第一個 layer 的 $\mu$, $\sigma$，class 仍舊可以透過 $\mu$, $\sigma$ 來判別
        1. ![](https://i.imgur.com/L31Cmop.png =250x)
        2. ![](https://i.imgur.com/O2F1lIM.gif =150x) (positional normalization 示意圖不包含 Batch，維度 C, H, W)
    - 比較下表中 classification task 的 error rate，單純從 moments 來做分類 (PONO moments, 紅色) 已經比隨機亂猜 (Random Baseline, 灰色) 來得更好。如果把 moments 移掉 (PONO normalized, 藍色)，結果會比標準的 PONO (綠色) 還要更爛，所以 moments 其實是重要的 feature
        - ![](https://i.imgur.com/qUFjuRx.png =400x)
- 本篇論文的方法基於 positional normalization，既然 moments 代表 shape 和 style，那只要交換 moments 就能限制模型同時學習 a instance 的 feature dist 和 b instance 的 moments
    - ![](https://i.imgur.com/G4NgVJ5.png)

# Methodology

前面其實已經把本篇論文的核心給講完了，接著細探這篇的方法



