@echo off
@setlocal EnableDelayedExpansion
set confidence_threshold=0.5
set in_dir=data\video
set wait_time=1
scripts\build\examples\yolo\Release\yolo models\yolo\MobileNetYOLO_deploy.prototxt ^
models\yolo\MobileNetYOLO_deploy.caffemodel ^
%in_dir% ^
-file_type video ^
-mean_value 0.5,0.5,0.5 ^
-normalize_value 0.007843 ^
-confidence_threshold !confidence_threshold! ^
-wait_time !wait_time! 

popd
@endlocal