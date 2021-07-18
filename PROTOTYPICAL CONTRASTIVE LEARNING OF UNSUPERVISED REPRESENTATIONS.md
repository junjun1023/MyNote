# PROTOTYPICAL CONTRASTIVE LEARNING OF UNSUPERVISED REPRESENTATIONS

- ICLR 2021 poster
- [arxiv](https://arxiv.org/abs/2005.04966), [open review](https://openreview.net/forum?id=KmykpuSrjcq)
- [Github](https://github.com/salesforce/PCL)

# Overview

- Instance discrimination tasks 包含 1. image transformation 2. contrastive loss 兩個要素（我看起來就是 contrastive learning）
- 現有 instance discrimination 的方法大都有同一個缺點：缺乏 semantic 的資訊
    - ![](https://i.imgur.com/U5ATGQV.png =200x)
    - 如上圖，就算類別 label 同是馬，但是「包含人的馬」跟「不包含人的馬」


