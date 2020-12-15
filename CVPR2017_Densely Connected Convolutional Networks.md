# Densely Connected Convolutional Networks

Composite Function H<small>l</small>(.)
: BN + ReLU + 3x3conv

> feature maps 要 concate 需要 feature maps 的 size ( spatial size ) 相同
>
> 但是 convolutional networks down sampling 可以增加視野範圍
>
> 所以作者將整個網路分成 3 個 dense blocks，3 個 dense blocks 以 **transition layer** 連接

Transition layers
: BN + 1x1conv + 2x2 average pooling

