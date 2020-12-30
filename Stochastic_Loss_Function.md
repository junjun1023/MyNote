# Stochastic Loss Function

## Concept
![](https://i.imgur.com/vd7hlV6.png)

簡單來說，用decision network組出一loss function，和main network算loss
這個loss會更新main network，再更新decision network

而decision network就是要學怎麼組出好的loss function


## Loss sampling
未知Bag of Loss的分佈，使用Gumbel Softmax Trick來對此Bag進行採樣
![](https://i.imgur.com/vbpPi0C.png)


ai => 選不選這個loss
bi => 這個loss的weighting

decision network就是要**學ai和bi**
為什麼要分成ai和bi，論文中好像說是ai是用來隔離掉noise loss function的

這個離散的公式是可以用RL來學ai和bi的，但作者認為RL必須要Main network validate acc時才可以更新，他們不想要有這個延遲

因此把上面的公式變成
![](https://i.imgur.com/JcDyn75.png)
其中
![](https://i.imgur.com/QQYp2OE.png)
然後B()是個Binary function

## Finally, stochastic loss function
![](https://i.imgur.com/AMVuYPu.png)

再來就是要讓這個loss可微分又保有好的gradient


## Gumbel-Softmax
詳細可參考 https://lovecoding.fun/article/what-is-gumbel-softmax-trick/

**什麼是categorical distriution**
![wiki](https://i.imgur.com/xbmgoYX.png)

從categorical distribution採樣的問題 => 若只取Max會有 discrete 不可微分的問題
因此使用Softmax，但這只解決了可微分的問題
採樣所需的隨機性 => Gumbel


## The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
reference : https://arxiv.org/abs/1611.00712
log(ak) is log probability of discrete RV

![](https://i.imgur.com/76P4Jtc.png)

where Gk is
![](https://i.imgur.com/ecSgDqE.png)

which is a Gumbel distribution

![](https://i.imgur.com/QkG7qJf.png)

## Gumbel Distribution
![](https://i.imgur.com/h8d1usL.png)
![](https://i.imgur.com/yJiAxzj.png)


## Gumbel-Max
![](https://i.imgur.com/4WHxWVQ.png)
## Gumbel-Softmax
![](https://i.imgur.com/d4IgU4V.png)

tau越小(接近0) => one hot, tau越大 => uniform

## Code
![](https://i.imgur.com/6Dmr5ge.png)
![](https://i.imgur.com/W8LhMJ9.png)
Softmax結果會都相同

![](https://i.imgur.com/sk8RenM.png)
![](https://i.imgur.com/OOygOaf.png)
gumbel softmax　有隨機性

![](https://i.imgur.com/sZcYR6c.png)
![](https://i.imgur.com/q7NERg3.png)
tau越小(接近0) => one hot, tau越大 => uniform

![](https://i.imgur.com/f8a1FsE.png)
gumbel-max => one hot

## How Stochastic loss function do sampling
![](https://i.imgur.com/Xjm4Py4.png)
![](https://i.imgur.com/v4x2e5t.png)
寫成數學就是
![](https://i.imgur.com/2zf10RG.png)

## What is best value for K
![](https://i.imgur.com/ubtIBHb.png)
可以看到根據不同資料集，最好的K不一定會一樣
所以可以證明Gumbel Softmax採樣的方法是比只用Softmax好的
(因為Softmax就是K=無限的Case)

## Algorithm
![](https://i.imgur.com/C4wXhaO.png)
