@echo off
@setlocal EnableDelayedExpansion

scripts\build\tools\Release\caffe train --solver=examples\yolo\darknet_v3\gnet_region_solver_darknet_v3.prototxt --weights=examples\conv18.caffemodel

popd
@endlocal