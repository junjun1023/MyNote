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
        - e<small>tj</small> 是 alignment model，衡量位置 j 附近的 input 和位置 t 附近的 output 有多匹配
        - 


## Data

### Training Data

兩種 Datasets，各有 300,000 個 topic 和 label 的 pair

- ds_wiki_tfidf
    - label: article title
    - topic: 文章中 TFIDF top30 的 terms
- ds_wiki_sent
    - label: article title
    - topic: 文章前 30 個 terms
- training ( 226,282 )，validate ( 12,424 )，test ( 11,800 )
- 移除數字、特殊字元、稀有字、stop words
- article titles ( labels ) 有 13,947 個 unique words
- ds_wiki_tfidf 有 181,793 個 unique words
- ds_wiki_sent 有 87,446 個 unique words

### Testing Data

兩個 Datasets 各自有 golden-standard labels

- ds_wiki_tfidf
    - topics_bhstia
    - 每個 topic 有 19 個 candidate labels
    - topics' top 10 terms
- ds_wiki_sent
    - topics_bhatia_tfidf 是 topics_bhatia 的 extend version，每個 topic 多新增 20 個 terms



## Experiments

### Hyper Params

隨機 random 看看哪個 validation loss 比較低選哪個

- optimizer: adam
- learning rate: 0.001
- loss: sparse categorical cross entropy loss
- encoder: BiGRU, 200 hidden unit
- decoder: GRU, 200 hidden unit
- dropout: 0.1

### Baseline

- top-2 terms and top-3 terms


### Label Evaluation

> BERTScore
> ![](https://i.imgur.com/VZCMO9n.png)
> 不需要 exact match，可以做到抽換同義詞

model's overall score is mean score over all topics
![](https://i.imgur.com/akFTSVb.png)


## Result and Discussion

- 用 ds_wiki_sent training，用 topics_bhatia_tfidf testing 的 BERTScore 最高，因為額外多增加的 20 個 terms


## LDA ( Latent Dirichlet Allocation )

- Topic Model 文件主題模型
- 每篇文件由數個不同比例的「主題 ( Topic )」組合而成
- 每個主題又可以用數個「用詞 ( Word )」組合而成，且相同的用詞可同時出現在不同的主題之間
- Topic Model 的目的就是透過文件找到背後隱含主題的結構


##### Reference
- https://medium.com/@tengyuanchang/%E7%9B%B4%E8%A7%80%E7%90%86%E8%A7%A3-lda-latent-dirichlet-allocation-%E8%88%87%E6%96%87%E4%BB%B6%E4%B8%BB%E9%A1%8C%E6%A8%A1%E5%9E%8B-ab4f26c27184