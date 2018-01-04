# caffe-yolov2

## Reference

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
> cd build
> Caffe.sln
> select release
> build solutions
```

## Usage

### convert model

`cd examples`

1. convert yolo.cfg to yolo.prototxt
2. convert yolo weights to caffemodel

### detection

`cd caffe_root`

1. examples\yolo_detection.cmd


### trainning

comming soon

Known issue 

1. box_data_layer can not read lable from lmdb
