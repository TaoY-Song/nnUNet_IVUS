@echo off
echo Starting test per epoch time...

nnUNetv2_train 1 3d_fullres_bs1 0 -tr nnUNetTrainerBenchmark_5epochs
nnUNetv2_train 1 3d_fullres_bs1 0 -tr nnUNetTrainerBenchmark_5epochs_noDataLoading

echo Test completed.