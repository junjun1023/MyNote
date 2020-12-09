# Automatic Generation of Topic Labels

- 透過 seq2seq model，根據 input ( a sequence of terms ) 產生 output ( a sequence of terms )
    - 本篇論文的 input 是 topic terms
- seq2seq 由 2 個 RNN 組成，一個是 encoder 另一個是 decoder

## Method

![](https://i.imgur.com/0CR577J.png)

### Encoder
1. input: topic terms
2. 把 input 做 embedding 到 300 維
3. 往下丟給 bi-directional GRU
    - 一個方向讀順序的 input，另一個方向讀反序的 input
4. ![](https://i.imgur.com/2Ink7B4.png)
    - hf<small>t</small> 是 GRU forward 方向的 output
    - x<small>t</small> timestamp t 的 input
    - h<small>t-1</small> 上一個 timestamp t-1 的 hidden state
    - h<small>t</small> 是把 forward 和 backward concate 起來

### Decoder
5. ![](https://i.imgur.com/VJ0DqgD.png)
    - y<small>t-1</small> 上一個 timestamp t-1 的 predict output
    - s<small>t-1</small> 上一個 timestamp t-1 的 hidden state
    - c<small>t</small> 是由 encoder 最後一個 hidden state 計算得出的針對每個 target word 的 context vector
        - ![](https://i.imgur.com/HcDhExA.png)
        - 




## LDA ( Latent Dirichlet Allocation )

- Topic Model 文件主題模型
- 每篇文件由數個不同比例的「主題 ( Topic )」組合而成
- 每個主題又可以用數個「用詞 ( Word )」組合而成，且相同的用詞可同時出現在不同的主題之間
- Topic Model 的目的就是透過文件找到背後隱含主題的結構


##### Reference
- https://medium.com/@tengyuanchang/%E7%9B%B4%E8%A7%80%E7%90%86%E8%A7%A3-lda-latent-dirichlet-allocation-%E8%88%87%E6%96%87%E4%BB%B6%E4%B8%BB%E9%A1%8C%E6%A8%A1%E5%9E%8B-ab4f26c27184