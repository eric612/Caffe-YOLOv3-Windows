@echo off
@setlocal EnableDelayedExpansion
set confidence_threshold=0.25
set wait_time=0
set in_dir=models\yolo

scripts\build\examples\yolo\Release\yolo models\yolo\yolo.prototxt ^
models\yolo\yolo_12500.caffemodel ^
%in_dir% ^
-file_type image ^
-mean_value 0,0,0 ^
-normalize_value 0.003921568627451 ^
-confidence_threshold !confidence_threshold! ^
-wait_time !wait_time! 

 
popd
@endlocal