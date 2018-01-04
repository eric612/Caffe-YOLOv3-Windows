#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=./solver.prototxt
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_multifixed_iter_2000.solverstate
WEIGHTS=./yolo.caffemodel
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_pretrain_rectify_iter_120000.solverstate
$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS \
    | tee train_darknet_anchor.log #--weights=$WEIGHTS #--gpu=0,1

