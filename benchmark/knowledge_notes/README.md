# Knowledge notes

One short, task-specific note of domain or data knowledge for each of 14 tasks
(`T<id>.md`: tasks 1, 5, 8, 9, 11, 24, 25, 31, 34, 37, 41, 43, 44 and 50). They are
used in the knowledge-note diagnostic of the paper.

In the tier `H2+hint` the note of the task is added to the task text given to the model;
everything else is as in tier `H2`. `harness/run_cell.py` reads the note from this folder,
so the tier can only be run for tasks that have one:

    python3 harness/scheduler.py --exp knowledge_note --tiers H2+hint --archs SA \
        --models <model> --runs 1,2,3 --tasks 1,5,8,9,11,24,25,31,34,37,41,43,44,50

The recorded runs of the diagnostic are in `results/knowledge_note/`.
