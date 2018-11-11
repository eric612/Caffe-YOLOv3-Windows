# caffe-yolov3-windows

A caffe implementation of MobileNet-YOLO detection network , first train on COCO trainval35k then fine-tune on 07+12 , test on VOC2007

Network|mAP|Resolution|Download|NetScope|Inference time (GTX 1080)|Inference time (i5-4440)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
MobileNet-YOLOv3-Lite|0.747|320|[caffemodel](https://github.com/eric612/MobileNet-YOLO/tree/master/models/yolov3)|[graph](http://ethereon.github.io/netscope/#/gist/816d4d061c77d42246c5c9d49c4cbcf4)|6 ms|150 ms
MobileNet-YOLOv3-Lite|0.757|416|[caffemodel](https://github.com/eric612/MobileNet-YOLO/tree/master/models/yolov3)|[graph](http://ethereon.github.io/netscope/#/gist/816d4d061c77d42246c5c9d49c4cbcf4)|11 ms|280 ms

* the [benchmark](https://github.com/eric612/MobileNet-YOLO/tree/master/benchmark) of cpu performance on Tencent/ncnn  framework
* the deploy model was made by [merge_bn.py](https://github.com/chuanqi305/MobileNet-SSD/blob/master/merge_bn.py) , or you can try my custom [version](https://github.com/eric612/MobileNet-YOLO/tree/master/examples/merge_bn)
* bn_model download [here](https://drive.google.com/file/d/1jB-JvuoMlLHvAhefGCwLGh_oBldcsfW3/view?usp=sharing) 

## Linux Version

[MobileNet-YOLO](https://github.com/eric612/MobileNet-YOLO)

## Performance

Compare with [YOLO](https://pjreddie.com/darknet/yolo/) , (IOU 0.5)

Network|mAP|Weight size|Resolution|NetScope
:---:|:---:|:---:|:---:|:---:
[MobileNet-YOLOv3-Lite](https://github.com/eric612/MobileNet-YOLO/tree/master/models/yolov3_coco)|34.0*|[21.5 mb](https://drive.google.com/file/d/1bXZtB_wZBu1kOeagYtZgsjLq2CX0BGFD/view?usp=sharing)|320|[graph](http://ethereon.github.io/netscope/#/gist/b65f6b955e99c7d4c29a4b8008669f90)|
[MobileNet-YOLOv3-Lite](https://github.com/eric612/MobileNet-YOLO/tree/master/models/yolov3_coco)|37.3*|[21.5 mb](https://drive.google.com/file/d/1bXZtB_wZBu1kOeagYtZgsjLq2CX0BGFD/view?usp=sharing)|416|[graph](http://ethereon.github.io/netscope/#/gist/b65f6b955e99c7d4c29a4b8008669f90)|
[MobileNet-YOLOv3](https://github.com/eric612/MobileNet-YOLO/tree/master/models/yolov3_coco)|40.3*|[22.5 mb](https://drive.google.com/file/d/1G0FeQ7_ETc3zPn5HayhKi8Dz1-I5hU--/view?usp=sharing)|416|[graph](http://ethereon.github.io/netscope/#/gist/0ec45a4ca896553a20f9f9c16e80149f)|
YOLOv3-Tiny|33.1|33.8 mb|416

* (*) testdev-2015 server was closed , here use coco 2014 minival

### Oringinal darknet-yolov3

[Converter](https://github.com/eric612/MobileNet-YOLO/tree/master/models/darknet_yolov3) 

test on coco_minival_lmdb (IOU 0.5)

Network|mAP|Resolution|Download|NetScope|
:---:|:---:|:---:|:---:|:---:
yolov3|54.4|416|[caffemodel](https://drive.google.com/file/d/12nLE6GtmwZxDiulwdEmB3Ovj5xx18Nnh/view?usp=sharing)|[graph](http://ethereon.github.io/netscope/#/gist/59c75a50e5b91d6dd80a879df3cfaf55)
yolov3-spp|59.3|608|[caffemodel](https://drive.google.com/file/d/17b5wsR9tzbdrRnyL_iFEvofJ8jCmQ1ff/view?usp=sharing)|[graph](http://ethereon.github.io/netscope/#/gist/71edbfacf4d39c56f2d82cbcb739ae38)

## Other models

You can find non-depthwise convolution network here , [Yolo-Model-Zoo](https://github.com/eric612/Yolo-Model-Zoo)

network|mAP|resolution|macc|param|
:---:|:---:|:---:|:---:|:---:|
PVA-YOLOv3|0.703|416|2.55G|4.72M|
Pelee-YOLOv3|0.703|416|4.25G|3.85M|



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

## Reference

> https://github.com/weiliu89/caffe/tree/ssd

> https://pjreddie.com/darknet/yolo/

> https://github.com/gklz1982/caffe-yolov2

> https://github.com/duangenquan/YoloV2NCS

> https://github.com/eric612/Vehicle-Detection

> https://github.com/eric612/MobileNet-SSD-windows

## License and Citation


Please cite MobileNet-YOLO in your publications if it helps your research:

    @article{MobileNet-YOLO,
      Author = {eric612,Avisonic},
      Year = {2018}
    }
