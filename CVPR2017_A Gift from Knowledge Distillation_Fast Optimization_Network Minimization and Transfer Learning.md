# A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning

- CVPR 2017
- [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)
- [github not found]()


![](https://i.imgur.com/DtmMT53.png)

---

# Knowledge Distilling

> 簡單來說，再好的 model 如果很肥大，inference 的時候就會受到各種限制，記憶體、運算量等等問題
> 
> 希望可以用一個比較輕量的 model 來學習另一個比較肥大 model 已經學習到的東西
> 
> 很像人類社會中「老師教導學生」的概念，把知識濃縮後授予，即是知識蒸餾

看來看去 Knowledge Distilling 入門的經典代表作就是 [NIPS 2014] Hinton 的 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

## Distilling the Knowledge in a Neural Network

這篇論文最重要的一點就是提出 ***Soft Target*** 的概念

### Soft Target

#### Hard Target


| A    | B    | C    | D    |
| :--: | :--: | :--: | :--: |
| 10^-6^ | 0.9 | 0.1 | 10^-9^ |

- 以 classification 來說，ground truth 可以被視為 one hot encoding，只有一個是 1 其他都是 0，就是所謂的 *hard target*
- 或是經過 softmax 後，機率分布會呈現極端，勢必有某個極大值與其他極小值，這種數值與數值呈現壓倒性落差，也是一種 *hard target*
- hard target 只有明確分出對和不對


#### Soft Target

| A    | B    | C    | D    |
| :--: | :--: | :--: | :--: |
| 0.05 | 0.3 | 0.2 | 0.005 |

- 輸出的機率分布比較平坦，降低 label 的區別性，以獲得更多資訊，例如可以知道各類別的相異/相似程度
- 提供更多隱含資訊 ( dark knowledge )

#### Softmax

![](https://i.imgur.com/FRkJBiU.png)

- T = 1 是 softmax 的特例
- softmax 當整體輸入放大 N 倍，輸出會變得極端；反之整體縮小，輸出也會比較平滑

---



[圖源](https://www.cnblogs.com/lart/p/11505544.html)

![image alt](https://img-blog.csdnimg.cn/20190911113740615.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BfTGFyVA==,size_16,color_FFFFFF,t_70)


student 的 loss 由兩個 loss 組成

![](https://i.imgur.com/qtuNWnA.png)

1. 和 task ground truth 的 loss
2. 和 teacher soft target 的 loss

# FITNETS: HINTS FOR THIN DEEP NETS

> FITNETS 利用 model 的深度，train 一個比 teacher 深且瘦的 student model
> 
> model 的 depth 代表 layer 數；width 代表每層 layer 的 neuron 數
> 
> [ICLR 2015](https://arxiv.org/pdf/1412.6550.pdf)

因為 CVPR 2017 這篇一直跟 FITNETS 做比較，所以就稍微讀了一下


>　FITNETS 這篇在上一篇 KD 的基礎下，介紹 ***Hint-based training***
>
> 因為上一篇只是單純讓 student 學習 teacher 的 soft target，可以想像成直接讓學生學習老師最後的答案
> 
> 這篇利用 teacher 的深度，讓 student 學習 teacher 每層 layer 的 feature，可以想像成讓 student 學習 teacher 做題中間得出的每個中間答案


## Hint-based Training
![](https://i.imgur.com/wghTfDL.png)


### 名詞定義

hint layer
: 簡單來說就是 teacher hidden layer
: hint layer 的 output 叫做 hint

guided layer
: 簡單來說就是 student hidden layer
: guided layer 負責學習預測 hint layer 的 output (hint)

### Hint-based training loss
![](https://i.imgur.com/vfN1s2w.png)

- μ~h~ 和 ν~g~ 分別是 teacher/student 到 hint/guided layer 的 params W~Hint~ 和 W~guide~ 的巢狀 function
    - 就是用 weight 表示 x 的 function，function 的輸出就是 feature

- γ 是 regressor function，用來調整 student 的 width 好讓和 teacher 的 width 一致
    - 為了降低參數量，用 convolutional regressor 而不是 fully connected regressor
- L2 loss function

### Fitnet Stage-wise Training

![](https://i.imgur.com/wUm8NUm.png)

W~Hint~
: 是 teacher 從最初的 layer 到 hint layer 的 weight

W~Guided~ 
: 是 student 從最初的 layer 到 guided layer 的 weight

1. Hints Training
2. Knowledge Distillation


