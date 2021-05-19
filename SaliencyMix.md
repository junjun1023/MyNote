# SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization

###### tags: `Data Augmentation` `Regulization` `Domain Adaptation` `Domain Generalization`

- Accepted to ICLR 2021
- [Github](https://github.com/SaliencyMix/SaliencyMix)
- [paper](https://openreview.net/pdf?id=-M0QkvBGTTq)
- [HackMD](https://hackmd.io/@lEHmUoFNSfOem4UTt7O44g/rkR_05ftO)

---
## Motivation
- Data augmentation 能讓 model 更 generalized(泛化)，使 model 更 robust
- 像 `CutMix` 這類的 aug，隨機切下影像的 patch 後，像貼補丁一樣把 patch 和另一張影像混合，並依照面積比例來給 label
    - 保留 regional dropout 的優勢，強迫 model 學習 non-discriminative 的部分
    - ![](https://i.imgur.com/aX3HeGD.png =80x)
- 由於切下影像的 patch 是隨機的，patch(from source image) 所含的 information 可能不足以代表 source image 的 label，例如 patch 都是 background，有可能會誤導 model 學到不適合的 features
    - ![](https://i.imgur.com/kHhlhVF.jpg =500x)
- 本篇論文提出 ***SaliencyMix***，透過 saliency map 選出比較具有 information 的 patch，再補丁到 target image

---
## Methods

### Scheme
![](https://i.imgur.com/bwNWquX.png)

這篇論文的方法真的很簡單，就連取得 saliency map 也只需要兩行 code 就能解決，直接 call cv2 的內建 function 就好，
簡單說明一下這篇論文用到的 annotations

### Selection of the source patch
```python=
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(source_img)
```

$I_{vs} = f(I_{s})$
: 拿 source image 過 computeSaliency 得到 saliency map

```python=
maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
# np.unravel_index 會回傳在 array 中最大 element 的 index
i = maximum_indices[0]
j = maximum_indices[1]
```

$i, j = argmax(I_{vs})$
: 拿到 salient map 中，數值最大的 index

```python=
bbx1 = np.clip(i - cut_w // 2, 0, W)
bby1 = np.clip(j - cut_h // 2, 0, H)
bbx2 = np.clip(i + cut_w // 2, 0, W)
bby2 = np.clip(j + cut_h // 2, 0, H)
```

最後以 $(i, j)$ 為中心 crop 出 patch，要 crop 多少++比例++的 source image 取決於 $\lambda$ 的值，$\lambda \in \{0, 1\}$

嘛不過實際應用上，所謂的 crop 其實只是在一個跟 source image 同樣大小的 mask $M$ 上，把要 crop 的部分設為 1，其他設為 0

### Mixing the patches and labels

$I_a = M ⊙ I_s + M′ ⊙ I_t$
: 根據 crop mask $M$ 把 source image 和 target image 相加，得到 augmented image

$y_a = λy_t + (1 − λ)y_s$
: 最後的 label 是 source image label 和 target image label 的 weigted sum
: 這篇論文把這種 label 稱為 ==smooth label==，包含 $\lambda$ 比例的 source image 和 $1-\lambda$ 比例的 target image

## Experiments

###### Table 1
![](https://i.imgur.com/PZQ27Q3.png)

###### Table 2
![](https://i.imgur.com/c26tau6.png)

###### Table 3
![](https://i.imgur.com/3RXLCEX.png)


![](https://i.imgur.com/XwAtw0H.jpg)


![](https://i.imgur.com/861mZAW.png)


## Conclusions

1. 既然這種方法可以做在 raw input 上，那理論上也可以做在 feature map 上，而且還不需要過 opencv 的 function，直接取 feature map 的 maximum 的 index 就好
2. 有一定的可能性會把 target image 的 foreground 通通都遮住，按照本篇論文的說法，這也可能會誤導 model 的學習



