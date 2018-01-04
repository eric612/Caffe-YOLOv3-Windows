# caffe-yolov2

## Reference

> https://github.com/eric612/Vehicle-Detection

> https://github.com/eric612/MobileNet-SSD-windows

> https://github.com/gklz1982/caffe-yolov2

> https://github.com/duangenquan/YoloV2NCS

## modifications

1. add pre-trained model
2. fix bugs
3. windows support
4. vehicle detection

### Configuring and Building Caffe 

```
> cd $caffe_root/script
> build_win.cmd
```

## Usage

### convert model

`cd examples`

1. convert yolo.cfg to yolo.prototxt
2. convert yolo weights to caffemodel

### detection

`cd caffe_root`

1. examples\yolo_detection.cmd

If load success , you can see the image window like this 

![alt tag](predictions.jpg)

### trainning

comming soon

### Known issue 

1. Box_data_layer can not read lable from lmdb
2. The result x,y,w,h between caffe and darknet were correct , but confidence was a little difference 
3. The output size was different when max pooling layer at kernel size = 2 , stride = 1 , so i modify the original caffe code 
4. The output size was different when conv layer at kernel size = 1, to avoid this problem I set pad = 0 (prototxt)