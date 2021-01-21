# VAE ( Variational Auto Encoder )

> 李宏毅老師的影片筆記

- AE 無法控制 decode 回來的東西，VAE 解決 AE 的這個缺點（左）
- VAE 就是在 code 的某個點加上 noise，在 noise 範圍內的圖都要能被 decode 成滿月

![](https://i.imgur.com/E1NdY1K.png =300x)

- 在 noise 重疊的範圍能希望同時 reconstruct 回弦月和滿月，就可能會 reconstruct 成介於滿月和弦月的月相

![](https://i.imgur.com/guWBwnp.png =300x)


## 架構

### Basic Definition

![](https://i.imgur.com/qkvbuaS.png =400x)

$m$
: 原來 encoder 的 code

$c$
: 加上 noise 之後的 code

$\sigma$
: 加上的 noise 的 variance
: 取 exponential 確保一定為正
: $exp(\sigma)$ 是 variance

$e$
: 從 normal distribution sample 出來的值


- 如果只是單純要 minimize reconstruct error 的話，那 $\sigma$ ( 由 model 自己學的 ) 直接設成 0 就好
- 等於原本的 auto encoder
- $\sigma ( variance ) = 0$ 就不會有不同 image overlap 的情形，reconstruction error 最小

### Add Constraint to Variance

![](https://i.imgur.com/koWxUqJ.png =300x)

![](https://i.imgur.com/5eyv0ra.png =300x)

- 綠色的線是 $exp(\sigma)-(1+\sigma)$，就是**藍色的線-紅色的線**
- $\sigma=0$，過完 exponential 後，會有 variance 最低，$variance=exp(\sigma)=1$
- 意味著 $variance = 1$ 時會有 loss 最低，模型就不會只顧著 minimize reconstruction error


### Estimate P(x)

![](https://i.imgur.com/Wq2FsEq.png =400x)


- VAE 的主要目標就是為了estimate the probability distribution $P(x)$
- 透過 Gaussian Mixture Model 來 estimate


![](https://i.imgur.com/kzwPoV9.png =400x)


1. 假設今天有某個 $P(x)$ 的分佈可以透過很多個 gaussian models 來近似
2. 每個 gaussian model 的權重 weight 是 $P(m)$
3. 每個 gaussian model 會有自己的 $\mu, \sum$
4. 就可以從某個 gaussian model $m$ sample 出 data $x$，即 $x|m$
5. $P(x)=\sum_{m}P(m)P(x|m)$
    - $P(x|m)$ 在 gaussian dist. 下 sample $x$ 的機率
    - $P(m)$ gaussian dist. 的權重
    - $P(x)$ 可以視為 $x$ 被 sample 的機率的期望值 ( 事件$*$權重的加總 )
6. $x$ 其實就很像用 distribution 來表示 classification


> **VAE 其實就是 Gaussian Model 的 distributed 的版本**

![](https://i.imgur.com/PhrxG0u.png =400x)

- 假設 $z$ 是一個 normal distribution 的 vector，$z$ 的每個 dimension 都代表一個 attribute
- 對應上述，$z$ 就是前面提到 gaussian model 的 weight，只是這邊這個 weight 是可以用一個 normal distribution 的 vector 來表示
- 當 guassian 是由 gaussian distribution 產生的時候，就代表有無窮多個 gaussian 
- 在 sample $z$ 上的點 $x$ 時，即 $x|z$，一定會對應到一個 gaussian，至於對應到 gaussian 的 $mean, std$ 是由 Neural Network 決定的


![](https://i.imgur.com/vBW2gcu.png =400x)

### Maximum Likelihood
![](https://i.imgur.com/km5kGfu.png)

#### Decoder

![](https://i.imgur.com/ZIPvcZP.png)


Likelihood
: $L=\sum_{x}logP(x)$
: 在某個分佈下，某個 sample 發生的機率
: 透過 NN，近似 $latent\ space$ 的 $\mu, \sigma$

#### Encoder
![](https://i.imgur.com/C882UZT.png)


#### 數學推導

![](https://i.imgur.com/nTIqhfo.png =400x)

$3th$ 
: $q(z|x)$ 可以是任何的 dist. 因為 $dz$ 只和 $q(z|x)$ 有關

$5th$
: $\int_{x}q(z|x)log(\frac{P(z,x)}{q(z|x)})dz = \int_{x}q(z|x)log(\frac{P(x|z)P(z)}{q(z|x)})dz = -KL(q(z|x)||P(z,x)$

$4th$
: 加上 $KL\ divergence\ \geq 0$，所以有下界 $5th$


![](https://i.imgur.com/zmu8dBO.png =400x)

- 固定住 $P(x|z)$ 調整 $q(z|x)$，會讓 $L_b$ 一直上升，最後 KL divergence 會完全不見
- 讓 $L_b$ 越大，那 $likelihood$ 就會越大

為什麼 $P(x)$ 跟 $q(z|x)$ 無關？
: $P(x)$ 跟 $q(z|x)$ 無關，因為 $P$ 是 dist. 跟 $q$ 這個 dist. 本來就無關

為什麼可以透過 $q(z|x)\ maximize\ L_b$ ?


![](https://i.imgur.com/Y4qADfm.png =400x)
![](https://i.imgur.com/6JqIUpN.png =400x)

- $q$ 是一個 neural network 的 dist.，給定 $x$ 分佈，sample 某個 $z$ 的 $\mu, \sigma$
- $-KL(q(z|x)||P(z)) \leq 0$

為什麼要 $minimize KL(q(z|x)||P(z))$ ?
: 

![](https://i.imgur.com/LRGGLzW.png)

- $\int_{z}q(z|x)logP(x|z)dz=E_{q(z|x)}[logP(x|z)]$
- 視為 weighted sum
- 從 $q(z|x)$ sample data，給一個 $x$ 的時候，根據 $q(z|x)$ 來 sample data，要讓 $logP(x|z)$ 越大越好

為什麼要 $maximize \int_{z}q(z|x)logP(x|z)dz$ ?
: 


### Problems of VAE

![](https://i.imgur.com/ep9OGf4.png =400x)

- VAE 想要產生某張 image 跟 database 的 image 越像越好，沒有想過真的產生新的 image，或是只是把 database 的 image 做 linear combination

##### [HackMD](https://hackmd.io/@lEHmUoFNSfOem4UTt7O44g/BJ6dXhBku)

