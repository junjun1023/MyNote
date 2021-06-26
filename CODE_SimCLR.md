# SimCLR 拆解筆記

- 原 [Github](https://github.com/sthalles/SimCLR)

---
```python=
n_views
```
- n_views = 2 代表要跟 2 個 negative samples 比較
- n_views = 3 代表要跟 3 個 negative samples 比較
- **dataset** 會吐出 n_views 個 transformed batch

## SimCLR.info_nce_loss

注意，是計算 loss，所以 model 已經給出 predictions，維度是 ```[batch * n_views, out_features]```
不過就算是 pretrain encoder，維度也可以是 ```[batch * n_views, C, H, W]```，畢竟不影響內積算 similarity

### labels
```python=
labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
```
- 這邊真的超級迷，看超久才懂
- **dataset 會吐出 n_views 個 transformed batch**，先 cat 再丟給 model，代表要計算 nce_loss 的資料目前長 ```[batch * n_views, out_features]```
    - 要找到對於每個 n_views 的每個 batch 的 positive
- 假設```n_views=4, batch=3```
    - ![](https://i.imgur.com/yt4QNHe.png =400x)
    - 每一次 ```0, 1, 2``` 代表一個 n_views，共有 4 個 n_views，```0, 1, 2```是 iterate 一個 batch
    - 拆解看每個 row
        - ```what[0, :]``` 標示出**每個 n_views 的 batch 的第一張的 positive**
        - ```what[1, :]``` 標示出**每個 n_views 的 batch 的第二張的 positive**
        - ```what[2, :]``` 標示出**每個 n_views 的 batch 的第三張的 positive**
    - 拆解看每個 column
        - 理論像上面總共 3 個 row 就可以代表 n_views 的 positives
        - 但事實上只是拿第 1 個 n_views 當 postive
        - 所以 n_views 也要 iterate

- 假設```n_views=4, batch=2```
    - ![](https://i.imgur.com/ufjlaVS.png =400x)
- 總的來說就是，對於每一個 n_views 當 positive，batch 中的每張圖片 (iterate batch) 輪流當 positive


### similarity_matrix

```python=
features = F.normalize(features, dim=1)
```
- 看著看著突然很疑惑，為什麼需要做 normalize

```python=
similarity_matrix = torch.matmul(features, features.T)
# batch * n_view, batch * n_views
```
- ```[batch * n_views, out_features]``` 內積，會得到 ```[batch * n_view, batch * n_views]```，就是算兩兩的相似度
- 提醒一下，這個 similarity_matrix 就是主要拿來算 nce_loss 的 model predictions


```python=
mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
```
- ```labels.shape[0]```這邊就是個數字而已，下面的 code 也有出現，```labels.shape[0] == labels.shape[1]```
- ```labels```的 diagonal 因為是自己跟自己的 positive 所以要 masked 掉

```python=
positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
# select and combine multiple positives
negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)        
# select only the negatives the negatives
```

- 根據 labels 分別取得 positives 跟 negatives
- ```positives.size() == [batch * n_views, n_views-1]``` ，因為要扣掉自己跟自己的 positives，所以減 1 
- ```negatives.size() == [batch * n_views, batch * n_views -1 - n_views]```


```python=
logits = torch.cat([positives, negatives], dim=1)
logits = logits / self.args.temperature
```
- 分別取出 positives 跟 negatives 再 cat 回來，只是為了確保 positives 在 negatives 前面

```python=
labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
```

- 這邊我就真的看不太懂為什麼是這樣實作的了，看了看 issue，也是有人問一樣的問題
- 在參考的 github 中限制```n_views = 2```，因為他計算 loss 的方式是 ```CrossEntropy```
- 因為這個實作是透過 ```CrossEntropy``` 計算，把 positives 放在前面，代表第 0 個 class
- 不過另一個實作 SimCLR 的 [github](https://github.com/Spijkervet/SimCLR/tree/5b4bdc808dd29761fc585f368e916bc090f6c213) 好像就不是這樣算的，不過我沒有去讀就是



