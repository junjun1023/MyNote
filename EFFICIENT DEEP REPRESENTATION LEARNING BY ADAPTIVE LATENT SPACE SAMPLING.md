# EFFICIENT DEEP REPRESENTATION LEARNING BY ADAPTIVE LATENT SPACE SAMPLING

[![hackmd-github-sync-badge](https://hackmd.io/MK1u7OYhTiyJI0u6Q_kkbg/badge)](https://hackmd.io/MK1u7OYhTiyJI0u6Q_kkbg)


- [arxiv](https://arxiv.org/abs/2004.02757), [open-review](https://openreview.net/forum?id=Byl3HxBFwH)
- 同場加映 [Suggestive Labelling for Medical Image Analysis by Adaptive Latent Space Sampling ](https://openreview.net/forum?id=If6dqlBcI) 是這篇 paper 的 short version
- 本篇被 ICLR 2020 reject，short paper 那篇被 MIDL 2020 reject
- open-review 給的評論多為「沒有理論基礎」（不過我個人覺得概念滿有趣的）

---

# Overview

- Training 的時候都會需要大量的樣本（拗口，就是 input data, dataset 的意思啦），但並不是所有的 input data 對於 training model 是有幫助的，==有沒有可能找到那些足夠代表整個 dataset 的 subset，並只 annotate 這個 subset==，就可以降低 label 的需求
- Hardness-aware learning 的目標是「找出對於 training 最有貢獻的那些樣本」，[Smart Mining for Deep Metric Learning](https://arxiv.org/abs/1704.01285) 在 embedding space 下找出哪些 sample 在 training 時會有比較大的 gradient
- 本篇方法透過在 VAE 的 latent space 下做 sampling，再拿這些 samples 去 train model，雖然實作在 VAE 上，但是任何 generative model 應該都可以


---

# Methodolgy

方法總共分為兩階段：

(1) Train 一個 VAE model 

(2) 在 latent space 做 sampling，從 VAE.decoder 將 samples 過 decoder 後，這些 samples 會被用來 train 主要的 model，而這個主要 model 的 loss 則會 back-propogate 更新 samples

![](https://i.imgur.com/6eYgUQm.png)


整個架構看起來不難，重點在於怎麼做 sampling，以及 loss 是怎麼更新 samples

## Pipelines


作者提了兩個不同的 sampling 方法，對應有稍微不一樣的 training pipelines，但是流程大致如下
1. 在 latent space 做 sampling 取得 embedding set
2. ==embedding 經過 decoder== 後，embeddings 會還原成 data，得到 trainset
3. 把 trainset 丟給 model 做 training
4. 透過 training 的 loss 再從 latent space 取得 embeddings
5. 新的 embeddings 過 step 2 得到 data，加到原本的 trainset 繼續 iterate (step 3)

---

作者提出兩種 samplings

(1) **s**amplings by **n**earest **n**eighbors ($SNN$) 

(2) **s**amplings by **i**nterpolation ($SI$)


![](https://i.imgur.com/KDD0SjY.png)

:::info
示意圖解

- 藍色點是**實際存在**的 embedding，這個 embedding 是 datset 中某張 image 的
- 橘色點是**不實際存在**的 embedding，這個 embedding 不是 dataset 中的任何一 image 的
- 箭頭表示更新 sampling 的方向，從原本的 embedding 延箭頭方向更新會得到新的 embedding
:::


Samplings by Nearest Neighbors 
: 取得新 embedding 後，因為新 embedding 不能對應 dataset 的 image，所以就拿新 embedding 最近的 neighbor embedding 來用

Samplings by Interpolation
: 取得新 embedding 後，就直接拿新 embedding 來用 

---

這兩種 sampling 方法適用於不同狀況，在進入 algorithms 之前，先來試想一下：

因為取得新的 embedding 後，會過 decoder 把 embedding 還原成 image，但是 embedding 還原出來的影像 **labeling tool 可能無法 label**，為了確保 decode 回來的 image 可以 label，所以 $SNN$ 直接用距離最近的 image 的 embedding 來做 decode

> 雖然我這邊有點不太理解，那為什麼不直接拿原本的 image 做 training 就好？


## Sampling by Nearest Neighbor

![](https://i.imgur.com/La9nWGM.png)


:::success
Annotations
- $x$：image
- $p$：$Encoder(x)$，image 透過 encoder 到 latent space 的 embedding、vector，跟 $x$ 有對應關係
- $x'$：$Decoder(p)$，embedding 經過 decoder 轉換回來的 image
- $y$：Ground-truth，主要 task 的 label
- $D$：$x$ 的 dataset

這邊只重點翻譯幾個 annotation，方便下文閱讀理解

此方法適用於需要確保 labeling tool 可以產生 label
:::

1. 首先，有一個 dataset $D$，用這個 dataset $D$ 訓練一個 VAE


接下來，要先有第一個 iteration 的 samples，後續才能透過 training loss 更新 samples，為了確保可以 label：

> line 2

2. 在 dataset $D$ 上隨機取得 subset，將 subset $T^{(1)}$ 拿去 train model

> 6, 7

3. model train 好後，在 subset $T^{(1)}$ 做隨機採樣一些 $x$，將 $x$ 過 VAE.encoder 取得這些 $x$ 的 embedding $p$

> line 8

4. 將 $p$ 過 VAE.decoder 還原成 image $x'$ 後，丟給 model 計算 loss
    - Algorithm 寫成 $G(p)$，$G$ 是 decoder，是 generative model

> line 9, 10

5. 透過 loss 計算 $p$ 的 gradient，這個 gradient 就是更新 sample 的方向，取得 harder sample $p'$
    - ![](https://i.imgur.com/kBNsTVR.png =150x)
    - 新的 sample $p'$ 代表 'harder sample'
    - 「sample」代表 embedding

> line 11

4. 這些 harder sample $p'$，因為前述提及 labeling tool 需要 embedding 真實存在才能 label，就在 $p'$ 附近找到距離最近 **(nearest neighbor)** 的 $p$

> line 12

5. 再將 $p$ 過 decoder 得到 $x'$，代表 $T^{(2)}$
    - Algorithm 的寫法是 $G(p)$，$G$ 是 decoder，是 generative model


6. 回到 step 2.，只是這時候是在 $T^{(2)}\bigcup T^{(1)}$ 隨機取 subset 去 train model

## Sampling by Interpolation

![](https://i.imgur.com/O5ULyQi.png)

- 待補
- 不過應該很好想像吧(x


---
# My Conclusions

- 雖然號稱可以減少 label 的需求量，可是每一次迭代都需要多 label 一些東西，不能一口氣 label 完，感覺有點冗余