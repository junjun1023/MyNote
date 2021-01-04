# SSD: Single Shot MultiBox Detector

經典的one-stage object detector paper
1. using **convolution filter** to predict object classes and bbox in One Stage Object detector
1. using **default boxes (anchor box)** in One Stage Object detector
1. predict in **multi-scale feature maps**

# Framework
![](https://i.imgur.com/puo6Ca4.png)
**說明**
* Backbone: 論文中使用VGG16，不過基本上要用Resnet,Mobilenet都可以

* Object Detection: 
在6張大小不同的feature map上做predict，使用**4個或6個anchor box**
輸出是 
***(feature map大小)x(feature map大小)x(Anchor box數量x(boundingbox偏移+class類別數量+1個背景類別))***

* Compare with YOLOv1: YOLOv1只在一個大小的feature map上做predict

# Output (Prediction)
![](https://i.imgur.com/gndR1cJ.png)
**說明**
* feature map 上每一個點都預設幾個 Anchor box (此論文中稱為Default Box)
* 6張feature map(38x38, 19x19, 10x10, 5x5, 3x3, 1x1)上分別預設**4, 6, 6, 6, 4, 4個anchor box**
* 每個Anchor box都會predict出 **類別(classification)** 和 **BoundingBox偏移量** ，以上圖為例，**8x8的feature map總共有8x8x6個Anchor box**
* 一個圖片的物件可能被不同大小的feature map中抓出來，**大的feature map可以看到小物件，小的feature map可以看到大物件**
* ![](https://i.imgur.com/24Vqw1c.png)

# Loss
![](https://i.imgur.com/YhvLY92.png)
![](https://i.imgur.com/EZmDWJO.png)


## What is matched default box?
哪些anchor box要列入計算loss呢?
根據原文
![](https://i.imgur.com/90E6bvr.png)
 
## what is positive sample and negative sample
和**ground true算iou > 0.5**且**分類分對**的**anchor box**是positive sample，其他都是negative


## Bouning box Location loss
和**faster rcnn**的anchor box location loss相同
![](https://i.imgur.com/LlfRRTd.png)

## what is smooth l1
![](https://i.imgur.com/tNybZJw.png)

## classification loss
![](https://i.imgur.com/eKGVWY6.png)
Softmax + cross entropy
