#!/usr/bin/env sh

CAFFE_HOME=../../..

SOLVER=./gnet_region_solver_darknet_v3.prototxt
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_multifixed_iter_2000.solverstate
WEIGHTS=./gnet_yolo_region_darknet_v3_pretrain_iter_600000.caffemodel
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_pretrain_rectify_iter_120000.solverstate
$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS \
    --gpu=1 2>&1 | tee train_darknet_anchor.log #--weights=$WEIGHTS #--gpu=0,1

