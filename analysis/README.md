# Analysis

The scripts in this folder compute the numbers, tables and data plots of the paper from the
run records in `results/`. Each script reads `cell_result.json` (the run record) and
`cell_eval.json` (the verdict) of every run of the experiments it needs, prints plain-text
tables to standard output, and writes the same numbers to a file when `--json FILE` is given.
Nothing is written into the repository unless an output path is given.

Run the scripts from the repository root. They need Python 3.11 or later and `numpy`;
`figures.py` also needs `matplotlib`, and the dense part of `retrieval.py` needs
`sentence-transformers` and a local copy of the embedding model `BAAI/bge-large-en-v1.5`.
Locations are taken from `harness/paths.py`; set `GISCLAW_RESULTS` to read the records from
another directory.

## Scripts

| Script | Reproduces | Command |
|---|---|---|
| `main_results.py` | Share of main-experiment runs with a deliverable and share judged correct; accuracy per model over the four agent configurations; the table of main results (accuracy, 95% interval and deliverable rate for C0, SA H0, SA H1, SA H2, DA H2, and the paired differences H0 - C0, H1 - H0, H2 - H1, H2 - H0, DA - SA); the same differences over faithful replays only; differences between Claude and open-weight models; tasks solved in every run and the hardest tasks; one-shot runs that ended without code | `python3 analysis/main_results.py` |
| `ablation.py` | The ablation table: change in accuracy of the H2 single agent when the dataset description, the domain-knowledge field or one GIS-layer component is removed; the field ablations without tasks T25, T31 and T42; component ablations against H1 | `python3 analysis/ablation.py` |
| `perturbation.py` | The perturbation table (perturbed variants against the unperturbed rerun) and the comparison of the rerun with the main experiment | `python3 analysis/perturbation.py` |
| `knowledge_note.py` | Knowledge-note diagnostic: correct runs before and after per model, sign test, bootstrap of the mean change, and the two subsets reported in the text | `python3 analysis/knowledge_note.py` |
| `variation.py` | Share of task-configuration cells with mixed results per model | `python3 analysis/variation.py` |
| `cost_and_tokens.py` | The token-use table (median and interquartile range of input and output tokens per run); median run time and number of rounds per model and configuration; total run time per model | `python3 analysis/cost_and_tokens.py` |
| `retrieval.py` | Offline retrieval test of the knowledge notes with BM25, dense embeddings and their hybrid | `python3 analysis/retrieval.py --embedding-model /path/to/bge-large-en-v1.5` |
| `figures.py` | The data plots: main accuracy, ablation, knowledge-note diagnostic with retrieval ranks, run-to-run variation | `python3 analysis/figures.py --out /path/to/figures` |
| `stats.py` | Shared helpers (loading the records, bootstrap, sign test); imported by the other scripts | |

`retrieval.py` runs the BM25 part when no embedding model is given. The model location can
also be set with the environment variable `GISCLAW_EMBEDDING_MODEL`. `figures.py` writes PDF
files to `analysis/out/` when `--out` is not given; that directory is not part of the
repository.

## Conventions

- **Tasks.** Task 30 is only partially gradable and is left out of every accuracy figure,
  which leaves 49 fully gradable tasks (`stats.PARTIALLY_GRADABLE`). The total run time in
  `cost_and_tokens.py` covers all 50 tasks.
- **Accuracy.** A run is correct when `analytically_correct` in `cell_eval.json` is true. A
  run without a verdict counts as not correct.
- **Deliverable.** `deliverable_produced` in `cell_result.json`: the agent ended the run
  itself and saved at least one output file.
- **Intervals.** 95% percentile intervals of a task-level cluster bootstrap: tasks are
  resampled with replacement, and all runs of a drawn task are kept together. Differences
  between two conditions are paired, with the same resampled tasks for both. The number of
  replicates is 2,000; it is 10,000 for the perturbation test, the knowledge-note diagnostic
  and the field ablations restricted to 46 tasks.
- **Knowledge-note diagnostic.** The unit is a model-task combination. Changes are tested
  with an exact two-sided sign test, and the mean change is bootstrapped over combinations.
- **Tokens.** Input tokens are `prompt_tokens + cached_tokens + cache_write_tokens`; output
  tokens are `completion_tokens`, which include reasoning tokens.
- **Seeds.** All random seeds are fixed in the scripts, so repeated calls print the same
  numbers.

## Numbers of the paper that these scripts do not reproduce

The following results rest on material other than the two record files of each run:

- **API cost per run.** The cost estimates apply the providers' list prices to the token
  counts. The token counts are in the records; the prices are not part of the repository.
- **Figure fidelity.** The share of correct runs whose figure contradicts the run's own
  data, and the accuracy estimates derived from it, come from a manual review of sampled
  figures. The review labels are not part of the run records.
- **Sensitivity analysis for alternative readings of a convention.** The list of the 16
  runs concerned comes from a manual review and is not part of the run records.
- **Case studies.** The descriptions of individual runs (causes of errors, values obtained
  after changing a run's code) are based on the code and output files of those runs.
- **Model calls that reached the output limit, and how wrong runs ended.** These counts
  need the per-call logs and the output files of the runs; the records hold per-run totals.
- **Knowledge notes by kind of knowledge.** The grouping of the 14 notes into three kinds is
  given in the paper's table of notes; the before and after counts of every model-task
  combination are printed by `knowledge_note.py`.
- **Selection steps.** Which tasks entered the perturbation test and the knowledge-note
  diagnostic was decided before these experiments were run; the scripts analyse the runs
  of the selected tasks.
- **Other figures.** The per-task overview, the case-study figures and the schematic
  figures are not drawn by `figures.py`.
