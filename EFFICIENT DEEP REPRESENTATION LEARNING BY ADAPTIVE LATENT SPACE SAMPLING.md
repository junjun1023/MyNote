# EFFICIENT DEEP REPRESENTATION LEARNING BY ADAPTIVE LATENT SPACE SAMPLING

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
(2) 從 latent space 下做 sampling，從 VAE.decoder 將 samples 過 decoder 後，這些 samples 會被用來 train 主要的 model，而這個主要 model 的 loss 則會 back-propogate 更新 samples

![](https://i.imgur.com/6eYgUQm.png)


整個架構看起來不難，重點在於怎麼做 sampling，以及 loss 是怎麼影響 sampling

## Pipelines

:::info
這篇論文是
1. 在 latent space 做 sampling
2. 經過 decoder 後，得到 trainset
3. 把 trainset 丟給 model 做 training
4. 透過 training 的 loss 再取得 samples
5. 新的 samples 會加到原本的 samples 繼續 iterate (step 3)

註：samples 都是 embeddings
:::

1. 首先，有一個 dataset 稱為 ⅅ，這個 dataset 的 data 都是還沒有被 annotate 的，用這個 dataset ⅅ 訓練一個 VAE

:::warning
要先有第一個 iteration 的 samples，但是第一次 samples 不可能是 embedding 上的每個 dimension 都是隨機數


我個人推測一來沒辦法確定一個 embedding 確實能代表一筆 data，二來沒辦法確定 embeddings 是在 latent space 裡，那怎麼做才可以確保 embeddings 在 latent space 裡呢？


還有一個好用的東西可以利用，就是 VAE 的 encoder，可以用來確保 embedding 是真實存在於 latent space
:::


2. 在 dataset ⅅ 中隨機 sample 出一些 data，例如 𝕜 筆 data，這 𝕜 筆 data 透過 VAE.encoder 取得第一個 iteration 的 embeddings
3. 這些 embeddings 過 VAE.decoder 取得 trainset
4. trainset 透過 labeling tool 取得 labels 並餵給 model 做 training

## Sample Methods

作者提出兩種 samplings
: (1) **s**amplings by **n**earest **n**eighbors (SNN) (2) **s**amplings by **i**nterpolation (SI)
![](https://i.imgur.com/KDD0SjY.png)

