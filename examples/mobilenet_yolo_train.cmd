@echo off
@setlocal EnableDelayedExpansion

scripts\build\tools\Release\caffe train --solver=models\yolo\MobileNet\solver2.prototxt --weights=models\yolo\MobileNet\mobilenet_iter_73000.caffemodel

popd
@endlocal