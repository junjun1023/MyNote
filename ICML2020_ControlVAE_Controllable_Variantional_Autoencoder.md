# ControlVAE: Controllable Variational Autoencoder



- KL Divergence 對 VAE ( AE ) 的影響
    - KL divergence 太低會降低 output 的 diversity 
        - 意思就是生成一個跟 input 差不多的東西，但是通常實際應用是希望能生成差不多但是不一樣的東西
        - **KL vanish**
        - 對於 text generation 或 image generation 有影響
    - KL divergence 太高會影響 accuracy
        - 透過降低 KL divergence 來限制每個 data sample 通過 latent channel 的上限
        - 強迫 latent space 學習到的東西更獨立 ( 不是冗餘的 )，更好可以 disentangle
        - disentagle representation learning

    - 需要人為可以根據 task 調控 KL divergence ( VAE 的 objective function ) 
- 透過自動控制 KL Divergence ( 在 data reconstruction 的 accuracy 和 application-specific constraint )
- 本篇論文透過三個應用面向來佐證 ControlVAE 的效用
    1. Image Generation
    2. Dialog Generation
    3. Disentangled Representation Learning
- [Github](https://github.com/shj1987/ControlVAE-ICML2020)
- [PyTorch AE series](https://github.com/AntixK/PyTorch-VAE) 


## Preliminaries

由於本篇論文提出一個自動控制 KL divergence 的方法，簡單介紹前人控制 KL divergence 的方法

1. $\beta-VAE$，$\beta\ increase\ from \ 0\ to\ 1$，但是這個方法只是盲目的改變 $\beta$ 的值，沒有在 training 過程實際去 sample KL divergence
2. $\beta-VAE\ (\beta > 1)$ 有 high reconstruction error 的問題，因為 KL divergence 太大，所以 model 傾向優先滿足 KL divergence

如何在 training 的過程中適當地調配 KL divergence 所佔的權重是本篇研究的核心

### PID Controller

![](https://i.imgur.com/s1Lt1bS.png)

- 上圖是本篇論文自動控制 $\beta(t)$ 的方法
- 藉由控制系統的想法，透過 feedback 來更新 $\beta(t)$
- non-linear PID controller
    - 比例-積分-微分控制器
    - 由比例單元（Proportional)、積分單元（Integral）和微分單元（Derivative）組成
    - ![](https://upload.wikimedia.org/wikipedia/commons/4/40/Pid-feedback-nct-int-correct.png)
- 計算 $set\ point$ 和 $controlled\ variable$ 的差 ( $error\ rate$ )，然後往降低 $error\ rate$ 的方向更新，即更新 $\beta (t)$
- 因為 $derivative$ ( 導數 ) 代表訊號的斜率，因此當訊號有 noisy 時，$derivative$ 會對 noise 有更多的響應，因此在 ControlVAE 中不會使用 $derivative$

set point
: 希望 KL divergence 維持的值
: 這邊的 $set\ point$ 就是 $\beta(t)$
: ![](https://i.imgur.com/XT2O0Dn.png =400x)

### [VAE ( Variational Auto Encoder )](https://hackmd.io/@lEHmUoFNSfOem4UTt7O44g/BJ6dXhBku)

## ControlVAE Algorithm

1. $training\ step\ t$
2. $sample\ output\ KL\ divergence, \hat{v_{kl}}$
3. 拿 $\hat{v_{kl}}$ 和 $set\ point\ v_{kl}$ 算 $error\ rate$ ( 差 )
4. $lower\ bound=E_{q_{\phi}(z|x)}[logp_{\theta}(x|z)]-\beta(t)D_{KL}(q_{\phi}(z|x)||p(z))$
    - VAE 的 lower bound，KL divergence 項用 $\beta(t)$ 調控
5. 當 KL divergence 低於 set point，就降低 $\beta(t)$，減少 KL divergence 在 objective function 的 penalty
6. 如果 KL divergence 高於 set point，則調升 $\beta(t)$，讓 KL divergence 的 penalty 上升
7. PI control: $\beta(t)=\frac{K_p}{1+exp(e(t))} - K_i\sum_{j=0}^{t}e(j) + \beta_{min}$
    - $K_p, K_i$ 是常數
    - $\frac{K_p}{1+exp(e(t))}\ range\ from\ 0\ to\ K_p$
        - 當 $e(t)$ 很大且為正，KL divergence 小於 set point，這一項幾乎為 0
        - $\beta(t)$ 會調低，讓 KL divergence 能上升
        - 反之當 $e(t)$ 很大且為負，KL divergence 大於 set point，這一項會接近 $K_p$，
        - $\beta(t)$ 會調升，讓 KL divergence 能下降
    - $K_i\sum_{j=0}^{t}e(j)$
        - 把過去某個 period T 的 error rate 都加起來
        - 對於校正會越來越強，直到 error 變號
        - 當 error 維持正時，前面的負號確保整個項都在下降，$\beta(t)$ 會調低，讓 KL divergence 能上升
        - 當 error 維持負時，前面的負號確保整個項都在上升，$\beta(t)$ 會調升，讓 KL divergence 能下降
        - 當 error 接近 0 時，整個項接近維持不變，$\beta(t)$ 也不太變動，允許 controller 讓 KL divergence 等於 set point
    - $\beta_{min}$ 
        - 是 application-specific constant
        - 移動 $\beta(t)$ 可以變化的範圍


![](https://i.imgur.com/PQwrfI8.png)


## PI Parameter Tuning for ControlVAE

> Tuning $K_p, K_i$ 是主要的 challenge

> Tuning 這些超參數確保對 error 的反應足夠平滑以允許收斂

> $\beta(t)=\frac{K_p}{1+exp(e(t))} - K_i\sum_{j=0}^{t}e(j) + \beta_{min}$

### $K_p$

- 當 KL divergence 接近 0 的時候，會出現最大誤差 $e(t)$，假設 $set\ point=v_{kl}$ 的話，$e(t) \approx v_{kl}-0 = v_{kl}$
- 當 KL divergence 太小時，VAE 無法從 input data 學到有用的資訊，所以需要 assign $\beta(t)$ 一個很小的非負數，好讓 KL divergence 可以上升

上述式子忽略其他項的話，$\beta(t)=\frac{K_p}{1+exp(v_{kl})} \leq \epsilon$
: $\epsilon$ 是很小的數，這篇論文中 $\epsilon=0.001$

$K_p \leq \epsilon ({1+exp(v_{kl})})$
: 作者發現 $K_p=0.01$ 有最好的 performance 且滿足不等式

### $K_i$

- 當 KL divergence 遠大於 $set\ point$ 的時候，$e(t)$ 會是一個很大的負數，$\frac{K_p}{1+exp(v_{kl})}$ 會趨近於常數 $K_p$，如果這時候 $\beta(t)$ 無法使 KL divergence 下降的話
- $- K_i\sum_{j=0}^{t}e(j)$ 前面的負號確保當累加的負值越多時，$\beta(t)$ 會是上升的
- $K_i$ 值不應太大，作者提出介於 $10^{-3}\ to\ 10^{-4}$ 之間的 $K_i$ 值有利於穩定訓練，也不應太小，會減緩收斂的速度

### Set Point

- $set\ point$ 的給定非常大一部分視任務而定
- 當 $\beta_{min} \leq \beta(t) \leq \beta_{max}$，KL divergence 的
    - 上界就是 VAE 在 $\beta(t)=\beta_{min}$ 收斂時的 KL divergence，定義為 $V_{max}$
    - 下界是 VAE 在 $\beta(t)=\beta_{max}$ 收斂時的 KL divergence，定義為 $V_{min}$
    - $\beta(t)$ 越小，KL divergence 越大；$\beta(t)$ 越大，KL divergence 越小
- $set\ point$ 改變的範圍要在 $[V_{min}, V_{max}]$
- ControlVAE 的 training 是 end-to-end 的，$set\ point$ 的值可以視任務需求決定
    - 需要多樣性高一點則 $set\ point\ (expected\ KL\ divergence)$ 高一點
    - 需要 accuracy 高則 $set\ point\ (expected\ KL\ divergence)$ 低一點

### Overview Algorithm

![](https://i.imgur.com/op9krvm.png =400x)

$10th\ to\ 11th$
: 是 PID/PI design 常用的限制，稱為 Anti-windup
: 當 Controller 給的 $\beta(t)$ 超出範圍時，控制了積分項的值，預防超出範圍的偏差

$14th\ to\ 19th$
: 限制 $\beta(t)$ 的範圍 $[\beta_{min}, \beta_{max}]$


## Applications of ControlVAE

### Language Modeling

- 用 ControlVAE 解決語言模型常碰到的 KL vanishing 同時增加生成數據的多樣性
- $K_p=0.01, K_i=0.0001, \beta(t)=[\beta_{min}=0, \beta_{max}=1]$

### Disentangling

- $\beta-VAE$ 將 $\beta$ 設為大於 1 的數，導致較大的 reconstruction error
- 透過每 K 個 steps 把 KL divergence 從 0.5 逐漸增加到預期的 $C$
- $K_p=0.01, K_i=0.0001, \beta_{min}=1, \because  \beta > 1$


### Image Generation

- 基本的 VAE 模型傾向於生成模糊且不切實際 image
- 與原始 VAE $\beta(t)=1$ 不同，$\beta(t)$ 介於 $[0, 1]$
- 給定 KL divergence，ControlVAE 可以在該範圍內自動調整 $\beta(t)$
- $K_p=0.01, K_i=0.0001, \beta(t)=[\beta_{min}=0, \beta_{max}=1]$


## Experiments

在前述 3 種不同的 tasks 上，透過 benchmark dataset 來 evaluate performance

- Language Modeling
    - Penn Tree Bank (PTB)
        - 42068 training sentences, 3370 validation sentences, 3761 testing sentences
    - Switchboard(SW)
        - 包含 2400 個 雙方的 telephone conversations
        - 隨機分割為 2316 training, 60 validation, 62 testing dialog
- Disentagling
    - 2D Shapes
        - 727,280 個 64x64 的二進制 2D 形狀圖像
        - 5 種 ground truth
        - shape(3), scale(6), orientation(40), x-position(32), y- position(32)
- Image Generation
    - CelebA(cropped version)
        - 202,599 張 128 x 128 x 3 的 celebrity images
        - 192,599 for training. 100,000 for testing


### Language Modeling

![](https://i.imgur.com/eVYKPWF.png)

> 跟另外兩個 baseline 做比較

#### Baselines

- Cost annealing
    - 透過 sigmoid 訓練 N steps 後再把 KL divergence  的超參數逐漸從 0 到 1
- Cyclical annealing
    - 將訓練過程分為M個週期，每個週期都使用線性函數將超參數從 0 增加到 1


$max\ error, KL\ divergence→0$