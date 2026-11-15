# Perturbation test

Ten tasks of the benchmark are varied in up to three ways, to see whether results depend
on the exact form of a task. The base tasks are 2, 10, 18, 19, 20, 24, 26, 29, 42 and 49.

| Perturbation | Variant id | What changes |
|---|---|---|
| Reprojected inputs | 100 + base id | Every vector layer the reference solution reads is written in another coordinate reference system (geographic layers in the local UTM zone, projected layers in EPSG:4326; both layers of task 24 in UTM zone 10N). |
| Renamed fields | 200 + base id | The attribute fields the analysis uses are renamed in the data and in the task text. |
| Reworded instruction | 300 + base id | The instruction is rephrased; data, checks and the rest of the task text are unchanged. |

Tasks 10, 20 and 24 have no renamed-fields variant, because their analysis uses no
attribute field. This gives 27 variants: 102-149, 202-249 and 302-349.

## Files

- `variants.json`: for every variant its base task and what is varied: the layers and
  their coordinate reference systems (`from`, `to`), or the renamed fields per input file.
  The three renamed-fields entries without a variant carry the reason (`not_applicable`).
- `reworded_instructions.json`: the reworded instruction per base task.
- `tasks.csv`: the task rows of the 27 variants, in the format of `benchmark/tasks.csv`.
  It is written by `harness/make_perturbations.py`.

## Building the variants

The inputs and the specs of the variants are not stored in the repository; they are
derived from the base tasks. With the base inputs in place (`harness/prepare_inputs.py`)
and the GDAL command-line tool `ogr2ogr` installed:

    python3 harness/make_perturbations.py

This writes `benchmark/inputs/<id>/dataset` and `benchmark/specs/<id>/` for every variant
and rewrites `tasks.csv`. With `--skip-inputs` only the specs and the task table are
written. The docstring of the script describes how the reference solution, the checks and
the control solutions of a base task are adapted so that every check keeps its meaning.
The recorded runs used inputs reprojected with GDAL 3.11; another GDAL version can write
coordinates that differ in the last digits.

## Running and scoring

A variant is handled like any other task once its inputs and spec exist. The gold is built
from the adapted reference solution, and the task text is taken from the variant table:

    python3 harness/build_all_gold.py --tasks 102,110,118,119,120,124,126,129,142,149,202,218,219,226,229,242,249,302,310,318,319,320,324,326,329,342,349
    python3 harness/scheduler.py --exp perturbation --task-table benchmark/perturbations/tasks.csv \
        --tiers H2 --archs SA --models <model> --runs 1,2,3 \
        --tasks 102,110,118,119,120,124,126,129,142,149,202,218,219,226,229,242,249,302,310,318,319,320,324,326,329,342,349
    python3 harness/score.py --exp perturbation

The recorded runs of the variants are in `results/perturbation/`. They are compared with
a rerun of the ten base tasks, recorded in `results/perturbation_rerun/`.
