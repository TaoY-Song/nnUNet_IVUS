@echo off

for /L %%i in (0,1,4) do (
    REM echo nnUNetv2_train 10 3d_fullres %%i --val --npz
    REM nnUNetv2_train 10 3d_fullres %%i --val --npz

    echo nnUNetv2_train 10 3d_fullres %%i --val
    nnUNetv2_train 10 3d_fullres %%i --val
)
REM nnUNetv2_find_best_configuration 10 -c 3d_fullres -f 0 1 2 3 4