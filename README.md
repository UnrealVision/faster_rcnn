# Faster RCNN

基于unreal_caffe的一个Faster-RCNN实现，你可能觉得这只是一个官方的修改版本，实际上并不是，我们要实现一个**纯C++的版本**!! 快star，keep an eye on it。 这个repo需要unreal_caffe来运行，所以你得先编译一下它，[传送们](https://github.com/UnrealVision/unreal_caffe). 说实话官方的faster-rcnn工程性不是非常好，我们这个项目的目的就是开源一个工程性极强的C++实现，both train and inference. 重点是，我们可能还会开源**TensorRT**的加速版本！！



# 安装

首先安装一下unreal_caffe, 然后编译lib:

```shell
cd lib
make
sudo python setup.py install
```

最后我们还需要下载一下模型来做直接的预测。为了让同志们可以下载模型文件，百度云特意放了一个链接，[传送门](https://pan.baidu.com/s/1geYnLWv)。下载下来之后解压到`data`下面即可。一切准备就绪。

接着运行：

```
./scripts/demo.py
```

然后你就可以看到：

![](https://i.loli.net/2017/12/28/5a44ac88da40e.png)

这是最后一类的检测结果。



# 训练自己的数据

不仅仅是inference，后面后面会继续更训练自己的数据，所有模型的prototxt都放在models下面。



# Copyright

UnrealVision all rights reserved.

