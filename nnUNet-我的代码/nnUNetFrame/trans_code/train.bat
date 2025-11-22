@echo off
setlocal enabledelayedexpansion

for /L %%f in (0,1,4) do (
    if exist "fold_%%f_completed" (
        echo Fold %%f already completed, skipping.
        goto :next_fold
    )

    echo Running fold %%f without --c...
    :retry_without_c
    for /F "tokens=*" %%o in ('nnUNetv2_train 10 3d_fullres %%f 2^>^&1') do set "output=%%o"
    set "exit_code=%errorlevel%"

    if %exit_code% equ 0 (
        echo Fold %%f completed successfully.
        type nul > "fold_%%f_completed"
        goto :next_fold
    ) else (
        echo Fold %%f failed, checking output...
        echo %output% | findstr /C:"NaN detected, current_epoch=" >nul
        if %errorlevel% equ 0 (
            for /F "tokens=*" %%e in ('echo %output% ^| findstr /C:"current_epoch="') do (
                for /F "tokens=2 delims==" %%i in ("%%e") do set "current_epoch=%%i"
            )
            echo NaN detected at epoch !current_epoch!.

            if !current_epoch! gtr 10 (
                echo Epoch enough, retrying with --c...
                nnUNetv2_train 10 3d_fullres %%f --c
            ) else (
                echo Epoch not enough, retrying without --c...
                goto :retry_without_c
            )
        ) else (
            echo Unknown error, retrying without --c...
            goto :retry_without_c
        )
    )

    :next_fold
)

endlocal