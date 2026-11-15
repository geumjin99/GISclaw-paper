# GISclaw

**Looking for the desktop application? See [GISclaw](https://github.com/geumjin99/GISclaw).**

Code, task materials and run records for the paper

> GISclaw: An Open-Source LLM Agent System for Autonomous Multi-Step Geospatial Analysis

GISclaw is an agent system in which a large language model solves a geospatial analysis
task by writing and executing Python step by step in a persistent, isolated sandbox. The
repository contains the agent, the execution and scoring harness, the materials for the
50 tasks of GeoAnalystBench as they were used in the paper, the records of every run, and
the scripts that compute the paper's numbers from these records.

## Contents

| Path | Contents |
|---|---|
| `agent/` | The agent: single-agent loop, dual agent (planner, worker, verifier), one-shot baseline, prompts, tools and the components of the GIS layer |
| `harness/` | Sandbox container, runner and scheduler, replay-based scoring, construction of inputs and reference outputs, tests |
| `harness/docker/` | Definition of the execution image with all package versions pinned |
| `harness/serving/` | Commands that serve the two open-weight models with vLLM |
| `benchmark/tasks.csv` | The task table: the text given to the agent for each of the 50 tasks |
| `benchmark/specs/<id>/` | Per task: reference solution, correctness checks, control solutions, metadata |
| `benchmark/knowledge_notes/` | The task-specific notes of the knowledge-note diagnostic |
| `benchmark/perturbations/` | Definition of the task variants of the perturbation test |
| `benchmark/supplements/` | Sources of the input files that the benchmark's download lacks |
| `results/` | Run record and verdict of every run reported in the paper |
| `analysis/` | Scripts that reproduce the tables, numbers and data plots of the paper |

## Reproducing the paper's numbers

The run records are part of the repository, so the statistics can be recomputed without
running any model:

```
pip install numpy matplotlib
python3 analysis/main_results.py
python3 analysis/ablation.py
python3 analysis/perturbation.py
python3 analysis/knowledge_note.py
python3 analysis/variation.py
python3 analysis/cost_and_tokens.py
```

`analysis/README.md` maps each script to the tables and figures it reproduces and lists the
numbers that cannot be recomputed from the records alone.

## Running the system

Requirements: Linux, Docker, Python 3.11 or later.

```
pip install -r requirements.txt
docker build -t gisclaw-exec:1.0 harness/docker
```

**Inputs.** The input data of GeoAnalystBench are distributed by its authors
(<https://github.com/GeoDS/GeoAnalystBench>) and are not part of this repository. After
downloading them, build the read-only input tree:

```
python3 harness/prepare_inputs.py --source /path/to/GeoAnalystBench
```

For ten tasks the download lacks input files; `benchmark/supplements/README.md` names the
files and their public sources.

**Reference outputs.** Scoring compares a run with the outputs of the task's reference
solution. Build them, and validate the checks on the control solutions, with

```
python3 harness/build_all_gold.py
```

**Control solutions.** Each task has between 8 and 25 control solutions under
`benchmark/specs/<id>/controls/`, 726 in total: scripts named `correct_*` that the checks must
accept and scripts named `wrong_*` that they must reject. Some were written by hand; many are
adapted from code that the evaluated agents produced in runs, and keep the block-by-block
form, the exploratory output and the retries of that code, because the checks have to judge
such code as it occurs. Controls of this kind rely on the names the sandbox preloads and are
meant to be run through the harness, in the execution image:

```
python3 harness/run_controls.py --task 8
```

`controls_result.json` in each task directory records the outcome for every control
(`harness/build_all_gold.py` rewrites it). `harness/validate_contracts.py` and
`harness/fuzz_checks.py` test the checks against missing, empty and malformed outputs.

**Harness levels.** H0 is the bare agent loop, H1 adds the general agent mechanisms, and H2
adds the GIS components:

| Component | Function |
|---|---|
| G1 | Data-profile card for every loaded layer |
| G2 | Notes from guards on spatial operations |
| G3 | Checks of figures and deliverables |
| G4 | Notes explaining errors of GIS libraries |
| G5 | Analysis rules in the system prompt (`agent/gis_rules.md`) |
| G6 | Lookup of GIS library APIs |
| G7 | Record of the files each block read and wrote (kept in the run record) |

**Models.** `harness/config.yaml` lists the five models of the paper with the request
parameters that were sent. The Claude models are called through the Anthropic API; set the
environment variable `ANTHROPIC_API_KEY`. The open-weight models are served locally with
vLLM (`harness/serving/`); any model behind an OpenAI-compatible endpoint can be added to
the configuration.

**A single run and a batch.**

```
python3 harness/run_cell.py --exp demo --tier H2 --arch SA --model claude-sonnet-5-5 --run 1 --task 1
python3 harness/scheduler.py --exp demo --models claude-sonnet-5-5 --archs SA,DA --tiers H0,H1,H2 --runs 1,2,3 --tasks 1-50
```

Tiers: `C0` one-shot code generation; `H0` the agent loop alone; `H1` with the agent
mechanisms; `H2` with the GIS layer in addition. A suffix switches something off or on:
`H2-g3` (without component G3), `H2+nodd` and `H2+nodk` (without the dataset description or
the domain-knowledge field of the task), `H2+hint` (with the task's knowledge note).
Architectures: `SA` single agent, `DA` dual agent.

**Scoring.**

```
python3 harness/score.py --exp demo
```

Each run is re-executed in a fresh container, the data behind its outputs are captured,
and the task's checks compare them with the reference outputs. The docstring of
`harness/score.py` describes the procedure and the fields of the verdict.

**Perturbation test.** `python3 harness/make_perturbations.py` builds the task variants;
see `benchmark/perturbations/README.md`.

**Tests.**

```
python3 -m pytest -q harness/tests
```

## Run records

`results/<experiment>/<tier>/<architecture>/<model>/run<k>/T<id>/` holds two files per run:
`cell_result.json` (configuration, outputs, rounds, run time, token use) and
`cell_eval.json` (verdict, the result of every check, and the outcome of the replay).
`results/README.md` describes the experiments and the fields. The released records are a
reduced form of what `harness/run_cell.py` and `harness/score.py` write: fields that describe
the machine and the time of a run are left out. The complete outputs and
execution logs of the runs are available from the authors on request.

## Licence

The code and materials written for this work are released under the MIT licence
(`LICENSE`). The task table and a small number of reference and control solutions derive
from GeoAnalystBench (Zhang et al., 2025), which is distributed under the Apache License
2.0; see `NOTICE` and `LICENSE-APACHE-2.0`.
