# [筆記] Uncertainty-guided Model Generalization to Unseen Domains

[![hackmd-github-sync-badge](https://hackmd.io/3dx_y1WLT7OVpU7wxEOorQ/badge)](https://hackmd.io/3dx_y1WLT7OVpU7wxEOorQ)


- CVPR 2021
- [arxiv](https://arxiv.org/abs/2103.07531)
- Github is not found

---

# Overview

- 機器學習演算法的成功多數基於 train data 和 test data 是獨立同分佈 **(i.i.d)**，也就是 train data 和 test data 都是從**同一個**分佈抽樣出來的，train data 的分佈和 test data 的分佈會很像，所以當 train data 和 test data 的分佈不一樣，model 往往會有錯誤的 prediction
- Domain generalization 旨於解決 train data 和 test data 分佈不一樣的問題，好讓 model 能更加泛化 (generalized)，「混合兩個以上的 domain 來 training，並且將沒有參與 training 的 domain 用作 testing」是最為直觀的其中一種想法
- 現今 domain generalization 的方法大都著墨於在 training 的時候混合兩個以上的 domain 來訓練，最為棘手的狀況是「在 training 的時候只有一種 domain」，故本篇方法致力於解決==如何只透過一個 domain 實現 domain generalization==
- 這篇論文提出的方法架構如下圖，具體而言，是透過 uncertainty 分別在 latent space 和 label space 做擾動 (perturbation)
    - ![](https://i.imgur.com/xNF9to9.png =300x)
    - 所謂的不確定性 (uncertainty)，是用標準差 $\sigma$ 來衡量，而這篇論文提到的以 uncertainty 為主導的擾動 (perturbation)，其實就是透過 $(\mu, \sigma)$ 來控制多維高斯分佈 (multivirate Gaussian distribution)；另外，擾動 (perturbation) 其實就是 augmentation，這兩個詞會交替出現在全篇論文
    - 在 latent space 做擾動的目的是希望能透過擾動 latent feature $\mathcal{h}$ 得到不同 domain 的 latent feature $h^{+}$，作法是採樣 multivirate Gaussian 再加回原本的 latent feature $h$ 得到 perturbated latent feature $h'$
    - 此外，作者提出有別於線性插值 (interpolation) 的 label auementation，利用 latent feature augmentation 的 $(\mu, \sigma)$ 作為取得 label augmentation 所需的 Beta Dist. 參數 $(\alpha, \beta)$，

