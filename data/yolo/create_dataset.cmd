@echo off
@setlocal EnableDelayedExpansion

set CAFFE_ROOT=..\..
set ROOT_DIR=.\
set RESIZE_W=416
set RESIZE_H=416
set LABEL_FILE=%CAFFE_ROOT%\data\yolo\label_map.txt
set LIST_FILE=%CAFFE_ROOT%\data\yolo\trainval.txt
set LMDB_DIR=.\lmdb\trainval_lmdb
set SHUFFLE=false


%CAFFE_ROOT%\scripts\build\tools\Release\convert_box_data --resize_width=%RESIZE_W% --resize_height=%RESIZE_H% --label_file=%LABEL_FILE% %ROOT_DIR% %LIST_FILE% %LMDB_DIR% --encoded=true --encode_type=jpg --shuffle=%SHUFFLE%

:: 2007 test
set LIST_FILE=%CAFFE_ROOT%\data\yolo\test_2007.txt
set LMDB_DIR=.\lmdb\test2007_lmdb
set SHUFFLE=false

%CAFFE_ROOT%\scripts\build\tools\Release\convert_box_data --resize_width=%RESIZE_W% --resize_height=%RESIZE_H% ^
  --label_file=%LABEL_FILE% %ROOT_DIR% %LIST_FILE% %LMDB_DIR%  --encoded=true --encode_type=jpg --shuffle=%SHUFFLE%
popd
@endlocal