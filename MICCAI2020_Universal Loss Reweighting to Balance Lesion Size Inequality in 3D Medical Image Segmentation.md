# Universal Loss Reweighting to Balance Lesion Size Inequality in 3D Medical Image Segmentation

###### tags: `paper`

https://arxiv.org/abs/2007.10033
Accepted to MICCAI 2020

## Inverse Weighting
![](https://i.imgur.com/QqVrmje.png)

根據面積做reweighting
本篇提出的方法（右），是不管class是否相同，只要不相連，都會有不同的weight

## Method
![](https://i.imgur.com/x7k8iz2.png)

![](https://i.imgur.com/oyAcAKT.png)

計算出一個weight map，算Loss的時候每個pixel都會乘上Weight Map上的Weight

## Modified Loss Function
![](https://i.imgur.com/7WgqoH0.png)

介紹loss
https://blog.csdn.net/m0_37477175/article/details/83004746

### BCE Family

因為pixel wise的分類，因此容易受到正負樣本不均等的問題影響，因此有做reweighted會顯著提昇效能

#### Weight Cross Entropy
這篇reference的就是UNet原論文，根據pixel以及面積做Class的reweighting
https://arxiv.org/abs/1505.04597
![](https://i.imgur.com/j2m1klM.png)


### Dice Family
https://stats.stackexchange.com/questions/438494/what-is-the-intuition-behind-what-makes-dice-coefficient-handle-imbalanced-data

Dice類的因為是算面積的比例，因此大的Lesion和小的Lesion並不影響
因此在面對class imbalance上Dice表現較BCE好

#### GDL
Dice 在計算時，小的lesion的variance會比大的lesion大，同樣差一個pixel，兩個結果會差很多

GDL便是為了balance這點，也有根據lesion面積做reweighting

## eval metrics == FROC
![](https://i.imgur.com/rD5fxTI.png)



## Experiment
![](https://i.imgur.com/nfxG6wN.png)

![](https://i.imgur.com/bf2F36T.png)
