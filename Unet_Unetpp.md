# Unet / Unet++

**paper**
U-Net: Convolutional Networks for Biomedical
Image Segmentation
Unet: https://arxiv.org/abs/1505.04597
UNet++: A Nested U-Net Architecture
for Medical Image Segmentation
Unet++: https://arxiv.org/abs/1807.10165
Unet++ 作者解說: https://zhuanlan.zhihu.com/p/44958351

Unet神经网络为什么会在医学图像分割表现好
https://www.zhihu.com/question/269914775

# Unet
## what we can learn from it
1. use of data augmentation on medical image (因為medical image一般都很少）
2. contracting path (類似resnet的residual path)
3. 如何更好的分割邊界

## Architecture
![](https://i.imgur.com/00CUNYp.png)

其中
1. 灰色的箭頭拉過去是作**concat**
2. 縮小feature map大小是使用max pooling
3. 

## contracting path 的作用

左圖為在high level(淺層，Unet架構中左邊，解析度比較高），右圖為再low level(深層，Unet架構中右邊）
可以看到想要在low level作Segmentation，必須要借助high level的資訊

TODO: why output 2 channel
TODO: why use concate, why not use add

## Unet output
**p.s 現在的Unet已經不搞decoder feature map比較小這件事了，左右大小都相同**

![](https://i.imgur.com/xKz02BC.png)
> the segmentation map only
contains the pixels, for which the full context is available in the input image.
This strategy allows the seamless segmentation of arbitrarily large images by an
overlap-tile strategy (see Figure 2).

簡單來說，output只有mask，你要把他疊回原圖，因輸出feature map大小較小，多的地方直接按照原輸入，這種設計可以減少GPU memory

**更新：**
有另外一種說法表示Unet左右feature map大小不一是因為為了Convolution沒用padding
https://stats.stackexchange.com/questions/474904/overlap-tile-strategy-in-u-nets
https://www.zhihu.com/question/268331470



## Data augmentation
1. 資料少
2. 要學到對於變形，predict不變

主要用到的augmentation:
* shift
* rotation
* grayscale value variations
* random elastic deformations
https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
![](https://i.imgur.com/VbU3CjK.png)
論文提到random elastic deformations在**標記的資料少**的時候非常重要

* Drop-out layers at the end of the contracting path


Discriminative Unsupervised Feature Learning with Convolutional Neural Networks
https://papers.nips.cc/paper/2014/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf

## Loss function
上面是1個label
![](https://i.imgur.com/4L8Vt5l.png)

![](https://i.imgur.com/bm9ZNVT.png)
對output feature每一channel 進行 softmax
k : channel
x : location of each pixel
ak(x) : pixel x經過activation後出來的結果
![](https://i.imgur.com/yqSxmAa.png)
然後pixel wise去和ground truth mask進行cross entropy
其中w => weight map是讓某些pixel在training的重要性更重

weight map如何得到請看下方

## separation of touching objects of the same class
![](https://i.imgur.com/GnrBEhd.png)

從label mask看，縫隙越小，loss的weight越大
![](https://i.imgur.com/ou7eJcJ.png)
wc: 每個class的pixel frequency
d1: 到最近的border的距離
d2: 到第2近的border的距離

w0, Sigma => hyperparameter。論文設定為w0=10, sigma=5

## initial weight
![](https://i.imgur.com/w0BFUY6.png)
Tl;dr
取高斯分佈，標準差 sqrt(2/N)，其中N是輸入神經元數量。
例如：
3x3 x 64 channel => N = 9 * 64 = 576

# Unet++
看這篇 https://zhuanlan.zhihu.com/p/44958351

簡單來說
![](https://i.imgur.com/iTXHdb0.png)
* 他把backbone接很多層出來作predict （類似FPN)
* 而他backbone的feature map會接很多條出去 （類似Dense Net)
* 每層predict都有算loss (deep supervision)
* 可以在inference time剪支（下圖）
![](https://i.imgur.com/avboj2f.png)


# Some interesting questions
* importance skip connection
https://arxiv.org/pdf/1608.04117.pdf
* why encoder and decoder resolution are different
U-net的論文中提到 overlap-tile strategy
![](https://i.imgur.com/OALU2N2.png)
https://stats.stackexchange.com/questions/474904/overlap-tile-strategy-in-u-nets
https://www.zhihu.com/question/268331470
* concat or add
https://stackoverflow.com/questions/49164230/deep-neural-network-skip-connection-implemented-as-summation-vs-concatenation
和@安 討論，add感覺會破壞feature map，比較像是為了加深網路，但不加深記憶體消耗的方案
Concat後過1x1 conv使用的是比較完好的feature map，而用conv來merge的好處是conv是可以學的，壞處是消耗記憶體
* cross entropy vs dice
https://stats.stackexchange.com/questions/321460/dice-coefficient-loss-function-vs-cross-entropy
![](https://i.imgur.com/ftefjEi.png)

> I would recommend you to use Dice loss when faced with class imbalanced datasets
提到dice loss的論文
[V-Net: Fully Convolutional Neural Networks for
Volumetric Medical Image Segmentation]https://arxiv.org/pdf/1606.04797.pdf
* why unet is good in medical image
https://www.zhihu.com/question/269914775