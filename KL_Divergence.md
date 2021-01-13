# KL Divergence

- 用來驗證**兩個機率分佈差異性的指標**
- 差異越大，所需的額外資訊量越大
- 因為其驗證不同分佈差異性的特性，適合應用在 loss-function 的設計上，可以在 backpropagation 的過程中，衡量預測解果與真實資料的相似程度


離散 KL Distance
: $D_{KL}(P||Q)=-\displaystyle\sum_{i}^{}P(i)ln\dfrac{Q(i)}{P(i)}$
: $D_{KL}(P||Q)=\displaystyle\sum_{i}^{}P(i)ln\dfrac{P(i)}{Q(i)}$

P
: 數據的真實分佈

Q
: 數據的理論分佈、估計的模型分佈、P的近似分佈

特性
: KL Distance 不具有對稱性
: $D_{KL}(P||Q)\neq D_{KL}(Q||P)$ 從分布 $P$ 到 $Q$ 的距離通常並不等於從 $Q$ 到 $P$ 的距離
: 當 $P$ 和 $Q$ 完全重合，$D_{KL}=0$，從 $P$ 到 $Q$ 不需要額外的資訊量 

---

- 如果有兩個獨立的機率分布 $p(x)$ 和 $q(x)$ 同時對應到同一個隨機變數 $x$，也就是它們所在的空間是一樣的，則可以使用KL Divergence來測量這兩個分布的差異程度
- KL-Distance 描述的是用預測結果 $Q$ 分佈透過編碼轉換後可以用來表達 $P$ 分佈所需的編碼複雜度
- 假設今天預測結果跟 $P$ 一模一樣，顯然用 $Q$ 來表達 $P$ 是不需要經過任何的編碼轉換
- 但如果 $P$ 跟 $Q$ 之間的資訊亂度是很大的，要用 $Q$ 來表示 P，轉換的過程想必是非常複雜，**所需要的資訊量**也更大
- 以 encode 的複雜度來看的話，那麼 $P$ 轉換到 $Q$ 跟 $Q$ 轉換到 $P$ 這兩件事的複雜度就不見得會相等，有可能正推是比較容易的，反推搞不好就很複雜



# Mutual Information

- Mutual information (MI) 是兩個 random variables 的 mutual dependence
- 把觀察一個隨機變數可以獲得另一個隨機變數的**資訊量**量化
- 表示 $(X,Y)$ 的 joint distribution 和 $X, Y$ marginal distribution 的乘積**差異**有多大
    - $P(X\cap Y)$ 和 $P(X)P(Y)$ 的差異有多大
- MI 是 pointwise mutual information ( PMI ) 的期望值
    - PMI 指單一事件，MI 指所有可能事件的平均

Definition
: $I(X；Y)=D_{KL}(P_{(X,Y)}||P_X\otimes P_Y)$
: 當 $P_{(X,Y)}$ 和 $P_X\otimes P_Y$ 重合，$I(X；Y)=0$，代表 $X, Y$ 獨立 ( 觀察 $Y$ 的值不會告訴跟 $X$ 有關的資訊 )





###### tags: [HackMD](https://hackmd.io/@lEHmUoFNSfOem4UTt7O44g/SyecU0qCP)