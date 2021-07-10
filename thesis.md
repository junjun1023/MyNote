# 碩論

## Todo

### NCKUH
- [ ] Implement [Contrastive learning of global and local features for medical image segmentation with limited annotations](https://hackmd.io/@lEHmUoFNSfOem4UTt7O44g/HyAyPpAnO) on NCKUH
- [ ] Sample some no_nodule images and train with NCKUH

# Daily

#### 07-10-2021

目前做實驗下來的感覺有點雜論無章，所以還是稍微紀錄一下好了。

原本想說用整個 NIH dataset 和 contrastive learning 來 pretrain encoder，無奈礙於時間太久，資料集太大，所以只好先跑在 NCKUH 上，但是因為 NCKUH 目前所使用的影像都是確定有 nodule 的，理論上來說要加入 non_nodule 的影像才對。

那既然加入 non_nodule 的影像，也就是有 class 的概念了，考慮到碩論想做的，從 high-dimension space 做 sampling，如果只是單純的 contrastive learning，可能會讓不相似的 class 距離比較近，相似的 class 距離比較遠，導致 sample 到別的 class。這個 class 的概念來自 [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)

不過這就感覺和 [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242) 背道而馳，這兩篇都得好好讀一下，不過本人不太想變成 supervised ，這就太麻煩了又是需要 label 的事情，但細想也只需要 image-level 的 label，好像不是說很難，NIH 就好了

另外，有找到一篇 [EFFICIENT DEEP REPRESENTATION LEARNING BY ADAPTIVE LATENT SPACE SAMPLING](https://openreview.net/pdf?id=Byl3HxBFwH) ICLR reject 的 paper，他做的事情是在 VAE 的 latent space 下做 sampling，感覺可以參考一下他的做法

#### 07-05-2021

- [paper] [Segmentation with Multiple Acceptable Annotations: A Case Study of Myocardial Segmentation in Contrast Echocardiography](https://arxiv.org/pdf/2106.15597v1)
    - 光看標題秒想到之前戴揚有提過，可以跟很多 groundtruth 算 loss 
- [paper] [Self-Loop Uncertainty: A Novel Pseudo-Label for Semi-Supervised Medical Image Segmentation](https://arxiv.org/abs/2007.09854)
    - 被 self-loop 吸引進來的，畢竟前陣子我想到的某一個架構也很像 self-loop

#### 06-28-2021 ~ 07-04-2021
- 把 CL 用在 lung cancer project 上
    - [x] InfoNCE train end-to-end, Exp3x...
    - [x] Pretrain CL with NIH dataset
- [note] [Contrastive learning of global and local features for medical image segmentation with limited annotations](https://hackmd.io/@lEHmUoFNSfOem4UTt7O44g/HyAyPpAnO)
    - NIPS 2020
    - 雖然是做在 3D 影像，但是解決 CL 做 segmentation
    - seminar 投影片做到一半～
    - 感覺可能要想一下 2D 影像的 local 是啥，往 chx 的方向看一下別人做哪些前處理當作 local contrastive

#### 06-27-2021
- 拆解完成，coding 在 lung cancer project 上，並 debug

#### 06-26-2021
- 拆解 [SimCLR code](/b8wDcisJToimeoLqBRJmhQ) 之筆記

#### 06-25-2021

- 第一次正式報 proposal 給學長姐跟老師聽
- 芳毅和宗樺給的方向，往 Contrastive Learning + VAE 的概念 survey
- [paper] ICLR 2021 reject [Momentum Contrastive Autoencoder](https://openreview.net/forum?id=ep81NLpHeos)
    - [Hyperspherical Variational Auto-encoders](https://arxiv.org/abs/1804.00891)
    - [Spherical Latent Spaces for Stable Variational Autoencoders](https://arxiv.org/abs/1808.10805)


#### 06-07-2021

- [ ] SCG-Net implementation and debug [MLG]
    - loss 降不下去啊啊啊啊啊啊

#### 06-03-2021

- [ ] 用 contrastive learning 做不同肺癌影像的 embedding (續) [Lung Cancer Project] 
    - 一般看到的 contrastive learning，例如 MoCo，幾乎都是做在 encoder 上，這樣就有幾種訓練模型的可能性
    - 要不要 end-to-end，不的話就是單獨先 pretrain UNet 的 encoder
    - 而 two-stage 的 training 則會有幾種可能，是要把 embedding 直接過 decoder；還是要把 embedding concat unet.encoder(x)，不過這樣就有 2 個 model，一個 UNet 學 segmentation，另一個 encoder 學 embedding
    - 然後不管是 one-stage 還是 two-stage，如果是 embedding 直接過 decoder 的話，套上 UNet 的 skip-connection 架構，是應該只看最後一個 stage 要滿足 NCE Loss，還是 encoder 的每個 stage 都要滿足 NCE Loss
- [x] Survey test time training 或 online training
    - [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://arxiv.org/abs/1909.13231) [ICML 2020]
    - [Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726) [ICLR 2021 spotlight]
    - [OnlineAugment: Online Data Augmentation with Less Domain Knowledge](https://arxiv.org/abs/2007.09271) [ECCV 2020]

#### 06-02-2021
- [ ] 專論實作收尾 [專論]
- [x] 加入 NIH dataset 訓練 seg+cls [Lung Cancer Project]
- [ ] 實作 SCG-Net 在 lung nodule segmentation 上 [Lung Cancer Project]
- [x] 肺癌計畫[週報](https://drive.google.com/drive/folders/11eJ4LEt9P_4T9vMRYixFRuRVzJ4jGK0L?usp=sharing) [Lung Cancer Project]
- [ ] 用 contrastive learning 做不同肺癌影像的 embedding [Lung Cancer Project] 
    - 會想做這個是因為，想要做到「可以在 feed 給 segmentation model 前，做類似醫生平常會做的事情，調整亮度、對比度等來更好找到 nodule」
    - 本來就在往 online 去做 paper survey，剛好找到一篇 [online augmentation](https://openreview.net/forum?id=hSUnRw1boTWm)，內容大致上是多 train 一個負責做 augmentation 的 model，這個 model 做完 augmentation 後再餵給 segmentation model
    - 不過這其實不太符合醫生平常會做的前處理，因為平常醫生的前處理是動態的；但如果今天是 feed 給 model 的話，就會是一個固定的 augmentation
    - 轉念一想，那其實可以透過 contrastive learning，讓同一張影像做完各種 pixel-wise 的 augmentation 後都能得到相似的 embedding，那 segmentation model 就只需要透過那個 embedding 做 segmentation


#### 06-01-2021


- [x] 準備肺癌計畫[週報](https://drive.google.com/drive/folders/11eJ4LEt9P_4T9vMRYixFRuRVzJ4jGK0L?usp=sharing) [Lung Cancer Project]
- [x] 人工智慧競技 final project 簡報 [DSAI]
- [x] MLG paper presentation [MLG]

#### 0522-0531-2021

- [x] 寄信請教論文作者專論實作的問題 [專論]
- [x] MLG Hw3 [MLG]
- [x] [group meeting](https://docs.google.com/presentation/d/1vwvudYnON5JzBUUfgCwzS5ZGJOB6UrdNk3sxCAH6D1M/edit?usp=sharing) presentation


#### 05-21-2021
- [x] Add classification to decoder [Lung Cancer Project]
    - [Global Pooling Operation for Image Classification](/crzbVk7_Q2WK0Fq9LTHPgA)
    
- [x] Read [group meeting paper](https://openreview.net/pdf?id=HCSgyPUfeDj)

- [ ] Change segmentation model from UNet to FPN and add classification heads to each layer of FPN decoder ==[Lung Cancer Project]==
    - 讀 group meeting paper 時，作者提到，lower layer representation 有時候表現得比 last layer 好，因為 last layer 最直接影響到 loss，為了能優化 loss，可能會遺失一些東西
    - 就想到「每個 layer 都做一次 predict 就好啦」，秒想到 FPN，除了每個 decoder layer 都可以拿來算一次 segmentation loss 外，還可以加上 classification loss
    - 暫時不知道為什麼在醫療影像上，UNet 比 FPN 還普及，可能有些坑，但反正想到了就先記錄下來


#### 05-20-2021
- [x] Survey [group meeting paper](https://openreview.net/pdf?id=HCSgyPUfeDj) and prepare for presentation
- [x] Try instance normalization on decoder [Lung Cancer Project]


# Motivation
## Main 
模仿醫師在判別影像前，例如 chest xray，都會「動態」調整影像對比度、亮度、反白等等 (data augmentation)，再進行判斷

1. 訓練一個負責做 augmentation 的 model，當作 preprocessing module
    - 沒有符合「動態」，因為 inference 的時候 model 通常是 freeze 的，丟給模型一個 input，模型的 output 是固定的
    - 固定的 output 不好，因為醫師通常是某種調整看不出來，那調整看看另一種

- test time augmentation
    - 在 inference 的時候，做各種不同 augmentation 丟給 model，把各種 prediction 結果總合輸出一個 final prediction
    - 跟我想做的事情很像，但是我又不想要事先定義「要做哪些 augmentation」、「augmentation 的強度應該多少」
    - 那這樣還不如直接讓醫師來親自做 augmentation，就感覺某種程度上還是受到限制了

2. 換個角度，讓不同的對比度、亮度、顏色的==同一張==影像，都能 map 到一個很相近的空間，用整個該空間去做 downstream task (segmentation, classifictaion)，就能同時包含各種不同 augmentation 的影像了
    - 沒錯，就是 contrastive learning
    - augmentation 就是跟顏色有關的 pixelwise 的 augmentation

- [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/pdf/2005.10242.pdf)
    - 這篇還沒細看，先偷看知乎說啥
    - ![](https://i.imgur.com/MNT6Stc.png =300x)
    - 大體上，contrastive learning 會將 data map 到一個球上，相近的 data 在球上的距離比較近，不相似的 data 在球上的距離比較遠

- 那這其實跟我想做的事情不太像，就算用上面 augmentation 的方法，那也只是同一個 source 的 image 在球球的距離會比較近，到頭來還是只能拿一個 embedding 去做 downstream task

- 思考中⋯⋯
    - 有沒有辦法拿很多個 embedding，畢竟一個 augmented image 就會有自己的 embedding？ 
    - Sampling 球球附近的點？這樣是單純把 test time augmentation 的行為拉到模型的中間？
    - 或是有沒有可能訓練一個模型，給模型一個 input image，模型吐出來的就是專屬於那個 image 的 hypersphere，這樣就可以直接拿整個 hypersphere 去做 prediction？

- 又聯想到 VAE 的 disentangle，每個 dimension 就是 input 的一個變因，也許一個模型負責 map 到 hypersphere，另一個模型負責所謂的「變因 (augmentation)」
    - 如果要用 encoder/decoder 的架構，GAN 咧 ?
    - Contrastive learning + VAE/GAN ??
    - Flow-based generative model 說是一對一的 z 和 x [reference](https://zhuanlan.zhihu.com/p/116775904)

- Contrastive learning 本身就是讓模型學習 invariant 的部分，那麼對著 pixel 做的 augmentation ，模型學習到的應該會是 shape 相關的東西吧？
    - 稍微看了一下 contrastive learning，為啥 MoCo 要有兩個 encoder ??

---

## Sub

往 softlabel 的方向思考，但是什麼毛毛都還沒想到，卡住了啊啊啊啊啊啊

- softlabel
- wasserstein


# 排程

## 110 暑假

### 碩論

- [ ] 閱讀 Contrastive Learning 相關論文
- [ ] 閱讀 VAE 相關論文
- [ ] 尋找和我想法相似的 related work 並實作
- [ ] 嘗試結合 Contrastive Learning & VAE

### 肺癌計畫
- [ ] 實作 SimCLR
- [ ] 嘗試 Contrastive Learning 在肺癌計畫上並優化
- [ ] 把碩論方向的實作嘗試在計畫上
- [ ] 每兩週會議準備

### Lab 事項
- [ ] 新生訓練準備