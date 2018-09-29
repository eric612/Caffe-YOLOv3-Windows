# caffe-yolov3-windows

A caffe implementation of MobileNet-YOLO (YOLOv2 base) detection network, with pretrained weights on VOC0712 and mAP=0.709

Network|mAP|Resolution|Download|NetScope|
:---:|:---:|:---:|:---:|:---:
MobileNet-YOLO-Lite|0.675|416|[deploy](https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov2/mobilenet_yolo_lite_deploy_iter_62000.caffemodel)|[graph](http://ethereon.github.io/netscope/#/gist/11229dc092ef68d3b37f37ce4d9cdec8)
MobileNet-YOLOv3-Lite|0.737|416|[deploy](models/yolov3/)|[graph](http://ethereon.github.io/netscope/#/gist/f308433ad8ba69e5a4e36d02482f8829)|
MobileNet-YOLOv3-Lite|0.717|320|[deploy](models/yolov3/)|[graph](http://ethereon.github.io/netscope/#/gist/f308433ad8ba69e5a4e36d02482f8829)|

Note : 
>1. Training from linux version and test on windows version , the mAP of MobileNetYOLO-lite was 0.668<br>

## Performance

Compare with [YOLOv2](https://pjreddie.com/darknet/yolov2/) , I can't find yolov3 score on voc2007 currently 

Network|mAP|Weight size|Inference time (GTX 1080)|Inference time (i5-4440)
:---:|:---:|:---:|:---:|:---:
MobileNet-YOLOv3-Lite|0.717|20.3 mb|6 ms (320x320)|150 ms
MobileNet-YOLOv3-Lite|0.737|20.3 mb|11 ms (416x416)|280 ms
Tiny-YOLO|0.57|60.5 mb|N/A|N/A
YOLOv2|0.76|193 mb|N/A|N/A

Note :  the yolo_detection_output_layer not be optimization , and the deploy model was made by [merge_bn.py](https://github.com/chuanqi305/MobileNet-SSD/blob/master/merge_bn.py)

### Oringinal darknet-yolov3

[Converter](https://github.com/eric612/MobileNet-YOLO/tree/master/models/darknet_yolov3) 

mAP|Resolution|Download|NetScope|
:---:|:---:|:---:|:---:
53.9|416|[caffemodel](https://drive.google.com/file/d/12nLE6GtmwZxDiulwdEmB3Ovj5xx18Nnh/view?usp=sharing)|[graph](http://ethereon.github.io/netscope/#/gist/59c75a50e5b91d6dd80a879df3cfaf55)

test on coco_minival_lmdb (IOU 0.5)

## Other models

You can find non-depthwise convolution network here , [Yolo-Model-Zoo](https://github.com/eric612/Yolo-Model-Zoo)

network|mAP|resolution|macc|param|
:---:|:---:|:---:|:---:|:---:|
PVA-YOLOv3|0.703|416|2.55G|4.72M|
Pelee-YOLOv3|0.703|416|4.25G|3.85M|

## Linux Version

[MobileNet-YOLO](https://github.com/eric612/MobileNet-YOLO)

### Configuring and Building Caffe 

#### Requirements

 - Visual Studio 2013 or 2015
 - [CMake](https://cmake.org/) 3.4 or higher (Visual Studio and [Ninja](https://ninja-build.org/) generators are supported)
 - Anaconda 

The build step was the same as [MobileNet-SSD-windows](https://github.com/eric612/MobileNet-SSD-windows)
 
```
> cd $caffe_root
> script/build_win.cmd 
```

### Mobilenet-YOLO Demo

```
> cd $caffe_root/
> examples\demo_yolo_lite.cmd
```

If load success , you can see the image window like this 

![alt tag](00002.jpg)


### Trainning Prepare

Download [lmdb](https://drive.google.com/open?id=19pBP1NwomDvm43xxgDaRuj_X4KubwuCZ)

Unzip into $caffe_root/ 

Please check the path exist "$caffe_root\examples\VOC0712\VOC0712_trainval_lmdb"


### Trainning Mobilenet-YOLOv3
  
```
> cd $caffe_root/
> examples\train_yolov3_lite.cmd
```


### Future work 

1. COCO training and eval

## Reference

> https://github.com/eric612/Vehicle-Detection

> https://github.com/eric612/MobileNet-SSD-windows

> https://github.com/gklz1982/caffe-yolov2

> https://github.com/duangenquan/YoloV2NCS