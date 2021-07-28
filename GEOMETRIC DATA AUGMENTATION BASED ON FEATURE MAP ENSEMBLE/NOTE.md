# GEOMETRIC DATA AUGMENTATION BASED ON FEATURE MAP ENSEMBLE

[![hackmd-github-sync-badge](https://hackmd.io/5Y2BWB61SY2wNV7cxvqVTQ/badge)](https://hackmd.io/5Y2BWB61SY2wNV7cxvqVTQ)


- ICIP 2021
- [arxiv](https://arxiv.org/abs/2107.10524)
- Github is not found

---


# Overview

- CNN 在 geometric transformations (例如：旋轉) 的表現都會比較差，最常見的作法是在把 image 餵給 model 之前，做一些 geometric augmentations，好讓 model 對於這類影像能比較 robust，不過 augmentation 開的越強，training 的難度也會越高
- 這篇論文不希望高強度的 augmentation 出現在 training，所以基於 TTA 之上改良，並加入 maxout 的想法，提出的方法不需要更改模型架構，能得到比 TTA 更好的結果


# Methodology

本篇方法架構如下圖 (b)：

![](https://i.imgur.com/Ft8RQHj.png)



## TTA

原本的 TTA 在把影像餵進模型前會做各種 augmentations，最後將模型輸出的數個結果透過 mean/max 集成 (ensemble)，但是模型如果沒有特別針對 geometric augmentations 做 training 的話，就會影響模型預測的準確度

## Proposed

本篇方法是，geometric augmented images 作為 backbone $f$ 前部份 $f_F$ 的輸入得到 feature $\tilde{z}$，透過 inverse transform 將 $\tilde{z}$ 還原，以旋轉舉例的話就是將 feature $\tilde{z}$ 轉回 $0^{\circ}$ 得到 $z$，把 $z$ 和 $\tilde{z}$ 對著 position 集成後得到 $\widehat{z}$，再將 $\widehat{z}$ 輸入到 backbone 的剩下部份 $f_R$，最後只輸出一個預測

本篇方法和 TTA 的最大差別在於，集成 (ensemble) 的對象是 feature，並且透過 inverse transform，模型的剩餘部份 $f_R$ 有機會可以看到旋轉回來的樣子

