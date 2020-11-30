# A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning

- CVPR 2017
- [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)
- [github not found]()

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

![](https://i.imgur.com/FRkJBiU.png =200x)

- T = 1 是 softmax 的特例
- softmax 當整體輸入放大 N 倍，輸出會變得極端；反之整體縮小，輸出也會比較平滑

---



[圖源](https://www.cnblogs.com/lart/p/11505544.html)

![image alt](https://img-blog.csdnimg.cn/20190911113740615.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BfTGFyVA==,size_16,color_FFFFFF,t_70)



# FITNETS: HINTS FOR THIN DEEP NETS

> FITNETS 利用 model 的深度，train 一個比 teacher 深且瘦的 student model
> 
> model 的 depth 代表 layer 數；width 代表每層 layer 的 neuron 數
> 
> [ICLR 2015](https://arxiv.org/pdf/1412.6550.pdf)

因為 CVPR 2017 這篇一直跟 FITNETS 做比較，所以就稍微讀了一下

