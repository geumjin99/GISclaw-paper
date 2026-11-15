# Run records

One directory per run:

```
results/<experiment>/<tier>/<architecture>/<model>/run<k>/T<id>/
    cell_result.json   the run: configuration, outputs, rounds, run time, token use
    cell_eval.json     the verdict: every check, and the outcome of the replay
```

| Experiment | Runs | Contents |
|---|---|---|
| `main` | 3,000 | Five models; single agent at H0, H1 and H2 and dual agent at H2; three runs of each of the 50 tasks |
| `oneshot` | 750 | One-shot code generation (C0); five models, three runs per task |
| `ablation` | 3,000 | H2 single agent without the dataset description (`H2+nodd`) or the domain-knowledge field (`H2+nodk`), four models; without one of the components G1 to G6 (`H2-g1` ... `H2-g6`), two models |
| `perturbation` | 162 | The 27 task variants of the perturbation test, two models, three runs |
| `perturbation_rerun` | 60 | Unperturbed rerun of the ten base tasks of the perturbation test |
| `knowledge_note` | 66 | H2 single agent with the task's knowledge note (`H2+hint`), 22 model-task combinations, three runs |

Task ids of 100 and above are variants of the perturbation test: 100 times the kind of
perturbation plus the base task (`benchmark/perturbations/README.md`). Task 30 is only
partially gradable and is left out of the accuracy figures of the paper.

## cell_result.json

The released files are a reduced form of the records the harness writes: the fields `status`,
`error`, `image_id` and `finished_at` of `cell_result.json` are left out. Scoring a run with
`harness/score.py` needs the run's output files and therefore applies to new runs.

| Field | Meaning |
|---|---|
| `tier`, `arch`, `model`, `run`, `task` | The configuration of the run |
| `tier_config` | The switches of the tier: tools offered, agent mechanisms, components G1 to G7, task fields withheld |
| `config`, `model_config` | Limits and the request parameters sent to the model |
| `outputs`, `required_outputs`, `required_present` | Files the agent saved; files the instruction names; whether all of them are present |
| `emergency_files` | Files saved by the harness when a run ended without the agent finishing |
| `deliverable_produced` | The agent ended the run itself and saved at least one output file |
| `rounds`, `self_corrections`, `agent_stats` | Agent rounds and counters of the loop, including the components that fired |
| `elapsed_s`, `llm_time_s`, `agent_time_s`, `hit_deadline` | Wall time, time waiting for the model, the agent's own time, and whether the time budget ended the run |
| `llm_stats` | Model calls and token counts summed over the run |
| `sandbox_events`, `container_memory`, `inputs_intact` | Events of the container (timeouts, restarts, out-of-memory), its memory limit, and the integrity check of the inputs |

## cell_eval.json

| Field | Meaning |
|---|---|
| `analytically_correct` | The verdict: true when every required check passed. A run without a verdict counts as not correct |
| `checks` | Every check with its name, whether it is required, whether it passed, the measured value and the threshold |
| `replay` | The re-execution of the run's code: success, blocks executed, files written, whether they are the files of the original run |
| `replay_faithful`, `replay_unfaithful_reasons` | Whether the replay reproduced the original run block by block, and the differences if not |
| `correct_without_emergency` | For runs with emergency files: the verdict when these files are ignored |
| `scorer_error` | True when the scoring itself failed; the run then has no verdict (one run, in `perturbation_rerun`) |
| `status` | `ok` for every released record |

`replay_options.json` lists the runs that were replayed with a larger memory limit than the
default (`--memory`) and those replayed without the blocks that the container had killed
in the original run (`--skip-killed`).
