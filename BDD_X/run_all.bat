@echo off
REM BDD-X flow-head experiment: 3 ablations x 3 seeds, frame and text variants, tables.
cd /d %~dp0

python prepare_data.py || goto :err
python train.py --all --seeds 0 1 2 || goto :err
python compare.py || goto :err

REM single frame (CoVLA-style)
python train.py --ablation f1_img_state --n-frames 1 --seeds 0 1 2 || goto :err
python train.py --ablation f3_img_only --n-frames 1 --seeds 0 1 2 || goto :err
python compare.py --tag __1f-concat || goto :err

REM action-only and justification-only targets
for %%F in (action justification) do (
  python train.py --all --text-field %%F --seeds 0 1 2 || goto :err
  python compare.py --text-field %%F || goto :err
)
echo done
goto :eof
:err
echo FAILED - see the message above
exit /b 1
