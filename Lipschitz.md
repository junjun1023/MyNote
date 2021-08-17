# Lipschitz 連續

[![hackmd-github-sync-badge](https://hackmd.io/w4yXN2GAQ7yuX97BHVCBHQ/badge)](https://hackmd.io/w4yXN2GAQ7yuX97BHVCBHQ)

---


微分方程
: 包含未知數和未知數導數的方程式

Initial Value Problem (IVP)
: 微分方程的初始值問題，在滿足初始條件下解微分方程

$x^{\prime}=f(t, x), \quad x(\tau)=A$
: 前面是微分方程，後面是 $IVP$

## Lipschitz Condition

[Youtube Tutorial](https://youtu.be/Cnc83B3C2pY)


如果存在一個 $L \geq 0$ 使得 $f$ 滿足下方不等式，則 $f$ 在集合 $D$ 滿足 Lipschitz condition

$|f(t, u)-f(t, v)| \leq L|u-v|, \quad$ for all $(t, u),(t, v) \in D \quad (2)$

所以只要看函式 $f$ 和集合 $D$ 就能知道 $f$ 有沒有滿足 Lipschitz condition


### Lemma 1 

Let $D$ be the rectangle
$$
R:=\{(t, p): t \in[a, b],|p-A| \leq B\}
$$
or the infinite strip
$$
S:=\{(t, p): t \in[a, b],|p|<\infty\}
$$

If $f: D \rightarrow \mathbb{R}$ and $\partial f / \partial y$ exists, is continuous and there is some constant $K \geq 0$ such that
$$
\left|\frac{\partial f}{\partial y}(t, y)\right| \leq K, \quad \text { for all }(t, y) \in D
$$
then $(2)$ holds with $L=K$.