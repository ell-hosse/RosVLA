@echo off
REM Full DriveLM-nuScenes experiment: data prep, both families, all ablations,
REM three text fields, 3 seeds, then the comparison tables.
cd /d %~dp0

python prepare_data.py || goto :err
python train_wm.py --self-test || goto :err

REM default caption field
python train_wm.py --all --seeds 0 1 2 || goto :err
python train_direct.py --all --seeds 0 1 2 || goto :err

REM the other two text fields (no-text ablations are reused, only text ones re-run)
for %%F in (behavior caption_full) do (
  python train_wm.py --ablation b_vla_diffusion --text-field %%F --seeds 0 1 2 || goto :err
  python train_wm.py --ablation d_vla_gt_caption --text-field %%F --seeds 0 1 2 || goto :err
  python train_direct.py --ablation d1_flow_generated --text-field %%F --seeds 0 1 2 || goto :err
  python train_direct.py --ablation d2_gt_caption --text-field %%F --seeds 0 1 2 || goto :err
  python train_direct.py --ablation d4_flow_state_only --text-field %%F --seeds 0 1 2 || goto :err
)

for %%F in (caption behavior caption_full) do python compare.py --text-field %%F
if exist ..\drivelm_nus_imgs_val.zip python compare.py --split nusc_val
echo done
goto :eof
:err
echo FAILED - see the message above
exit /b 1
