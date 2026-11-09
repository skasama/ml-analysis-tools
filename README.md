# analysis_tools

入力ファイルを自動検索せず、コマンドラインで明示的に指定する構成です。相対パスは各 `--run-dir` を基準に解決されます。

## Apptainerでの利用

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Loss

```bash
python scripts/plot_loss.py \
  --run-dir /path/to/run \
  --train-log outputs/log_train_0.csv \
  --val-log outputs/log_val.csv \
  --output-dir plots/loss \
  --output-file loss_curve_readable.png
```

## 共通のrun出力

分類、vertex、directionでは次の形式です。

```bash
python scripts/plot_direction_regression.py \
  --run-dir /path/to/run \
  --predictions outputs/preds.npy \
  --targets outputs/targets.npy \
  --indices outputs/indices.npy \
  --output-dir plots/direction
```

`plot_classification.py`と`plot_vertex_regression.py`も同じ `--predictions`, `--targets`, `--indices` を受け取ります。

## 複数runのresolution

`plot_resolution_dependencies.py`では、各runに対して `--vertex-predictions`、`--direction-predictions`、`--energy-predictions` のように個別指定します。対応するtargetsとindicesも同様です。

## 3D event display

HDF5 dataset名も明示できます。

```bash
python scripts/save_event_display_3d.py \
  --h5file events.h5 \
  --geometry-file geometry.npz \
  --event-index-dataset event_hits_index \
  --pmt-dataset hit_pmt \
  --time-dataset hit_time \
  --charge-dataset hit_charge \
  --position-key position \
  --event 0 --channel charge --output plots/event0.png
```
