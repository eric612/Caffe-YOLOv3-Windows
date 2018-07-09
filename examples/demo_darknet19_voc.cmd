@echo off
@setlocal EnableDelayedExpansion
set detector=1
set confidence_threshold=0.3
set in_dir=data\
set wait_time=2000
build\examples\ssd\Release\ssd_detect models\convert\darknet19_voc_deploy.prototxt ^
models\convert\darknet19_voc_deploy.caffemodel ^
%in_dir% ^
-file_type image ^
-mean_value 0.0,0.0,0.0 ^
-normalize_value 0.0039138 ^
-confidence_threshold !confidence_threshold! ^
-wait_time !wait_time! 
echo INFO: ============================================================
echo INFO: file_type : !file_type!
echo INFO: mean_value : 0.0,0.0,0.0
echo INFO: normalize_value : 0.0039138



echo INFO: input_folder : !in_dir!
echo INFO: confidence_threshold : !confidence_threshold!
echo INFO: wait_time : !wait_time!
echo INFO: ============================================================

popd
@endlocal