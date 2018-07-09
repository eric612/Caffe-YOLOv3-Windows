# caffe-yolov2-windows

A caffe implementation of MobileNet-YOLO (YOLOv2 base) detection network, with pretrained weights on VOC0712 and mAP=0.709

Network|mAP|Download|Download|NetScope
:---:|:---:|:---:|:---:|:---:
MobileNet-YOLO-Lite|0.675|[train](models/MobileNet/mobilenet_iter_73000.caffemodel)|[deploy](https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov2/mobilenet_yolo_lite_deploy_iter_62000.caffemodel)|[graph](http://ethereon.github.io/netscope/#/gist/11229dc092ef68d3b37f37ce4d9cdec8)
MobileNet-YOLO|0.709|[train](models/MobileNet/mobilenet_iter_73000.caffemodel)|[deploy](https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov2/mobilenet_yolo_deploy_iter_80000.caffemodel)|[graph](http://ethereon.github.io/netscope/#/gist/52f298d84f8fa4ebb2bb94767fa6ca88)

Note : Training from linux version and test on windows version , the mAP of MobileNetYOLO-lite was 0.668 

## Performance

Compare with [YOLOv2](https://pjreddie.com/darknet/yolov2/)

Network|mAP|Weight size|Inference time (GTX 1080)
:---:|:---:|:---:|:---:
MobileNet-YOLO-Lite|0.675|16.8 mb|10 ms
MobileNet-YOLO|0.709|19.4 mb|24 ms
Tiny-YOLO|0.57|60.5 mb|N/A
YOLOv2|0.76|193 mb|N/A

Note : the yolo_detection_output_layer not be optimization , and batch norm and scale layer can merge into conv layer

## Linux Version

[MobileNet-YOLO](https://github.com/eric612/MobileNet-YOLO)

## Modifications

1. caffe training 
2. add pre-trained model
3. fix bugs
4. windows support

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

### Darknet YOLOv2 Demo (COCO)

Download [yolov2 coco weights](https://pjreddie.com/darknet/yolov2/)

Save at $caffe_root/models/convert 

cd $caffe_root/models/convert 

```
python weights_to_prototxt.py
```

Note : you may need specify python caffe path or copy python lib. here

cd $caffe_root

```
examples\demo_darknet19.cmd
```

### MobilenetYOLO Demo

Download [deploy model](https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov2/mobilenet_yolo_lite_deploy_iter_62000.caffemodel)

Save at $caffe_root/models/yolov2

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

Download [pre-trained weights](https://drive.google.com/file/d/141AVMm_h8nv3RpgylRyhUYb4w8rEguLM/view?usp=sharing) , and save at $caffe_root\model\convert

### Training Darknet YOLOv2 

```
> cd $caffe_root/
> examples\train_yolo_darknet.cmd
```


### Trainning MobilenetYOLO
  
```
> cd $caffe_root/
> examples\train_yolo.cmd
```


### Future work 

1. yolov3 

## Reference

> https://github.com/eric612/Vehicle-Detection

> https://github.com/eric612/MobileNet-SSD-windows

> https://github.com/gklz1982/caffe-yolov2

> https://github.com/duangenquan/YoloV2NCS