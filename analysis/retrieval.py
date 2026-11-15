#!/usr/bin/env python3
"""Offline retrieval test: can a task's knowledge note be found from the task text alone?

    python3 analysis/retrieval.py [--notes DIR] [--embedding-model PATH] [--json FILE]

The pool holds the knowledge notes (one file T<id>.md per task, by default in
benchmark/knowledge_notes) together with every sentence of the domain-knowledge field of
all tasks in benchmark/tasks.csv. The query for a task is its instruction followed by its
dataset description. For every task that has a note, the script reports the rank of that
note in the pool under three retrievers:
  * BM25 (k1 = 1.5, b = 0.75, lower-cased alphanumeric tokens);
  * dense retrieval with BAAI/bge-large-en-v1.5 (cosine similarity, with the model's
    query instruction);
  * a hybrid of the two by reciprocal rank fusion (k = 60).
Two further pools are reported for BM25 and dense retrieval: without the domain-knowledge
sentences of the query's own task, and with the notes only.

The dense retriever needs the sentence-transformers package and a local copy of the
embedding model, given with --embedding-model or the environment variable
GISCLAW_EMBEDDING_MODEL. Without it only the BM25 part is run.
"""
import argparse
import collections
import csv
import glob
import math
import os
import re

import numpy as np

import stats
from stats import paths

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
CUTOFFS = (1, 3, 5, 10)
POOLS = collections.OrderedDict([
    ("notes and all domain-knowledge sentences", lambda entry, task: True),
    ("without the domain-knowledge sentences of the query's task",
     lambda entry, task: not (entry[0] == "dk" and entry[1] == task)),
    ("notes only", lambda entry, task: entry[0] == "note"),
])


def read_notes(notes_dir):
    notes = {}
    for path in sorted(glob.glob(os.path.join(notes_dir, "*.md")) + glob.glob(os.path.join(notes_dir, "*.txt"))):
        m = re.match(r"T?(\d+)", os.path.basename(path))
        if m:
            with open(path, encoding="utf-8") as fh:
                notes[int(m.group(1))] = fh.read().strip()
    if not notes:
        raise SystemExit(f"no knowledge notes (T<id>.md) found in {notes_dir}")
    return notes


def read_tasks():
    with open(paths.TASKS_CSV, encoding="utf-8", newline="") as fh:
        return {int(r["id"]): r for r in csv.DictReader(fh)}


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if len(s.strip()) > 20]


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(query, docs, k1=1.5, b=0.75):
    docs = [tokens(d) for d in docs]
    n = len(docs)
    mean_len = sum(map(len, docs)) / n
    df = collections.Counter(w for d in docs for w in set(d))
    terms = set(tokens(query))
    scores = []
    for d in docs:
        tf = collections.Counter(d)
        s = 0.0
        for w in terms:
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / mean_len))
        scores.append(s)
    return np.array(scores)


class DenseRetriever:
    def __init__(self, model_path):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path)
        self._docs = {}

    def scores(self, query, docs):
        key = tuple(docs)
        if key not in self._docs:
            self._docs = {key: self.model.encode(docs, normalize_embeddings=True)}
        q = self.model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)[0]
        return self._docs[key] @ q


def rank_of(scores, pool, task):
    """1-based rank of the note of `task` when the pool is sorted by descending score."""
    order = np.argsort(-scores, kind="stable")
    return 1 + [pool[j][:2] for j in order].index(("note", task))


def fuse(score_lists, k=60):
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        for position, j in enumerate(np.argsort(-scores, kind="stable")):
            fused[j] += 1 / (k + position + 1)
    return fused


def compute(notes_dir=None, embedding_model=None):
    notes = read_notes(notes_dir or paths.NOTES)
    tasks = read_tasks()
    entries = [("note", t, text) for t, text in sorted(notes.items())]
    entries += [("dk", t, s) for t, row in sorted(tasks.items()) for s in sentences(row["Domain Knowledge"])]
    queries = {t: tasks[t]["Instruction"] + "\n" + tasks[t]["Dataset Description"] for t in sorted(notes)}
    dense = DenseRetriever(embedding_model) if embedding_model else None

    out = {"notes": len(notes), "pools": {}}
    for name, keep in POOLS.items():
        ranks = collections.defaultdict(dict)
        size = None
        for task, query in queries.items():
            pool = [e for e in entries if keep(e, task)]
            docs = [e[2] for e in pool]
            size = len(pool)
            scores = {"BM25": bm25_scores(query, docs)}
            if dense:
                scores["dense"] = dense.scores(query, docs)
                scores["hybrid"] = fuse([scores["BM25"], scores["dense"]])
            for method, sc in scores.items():
                ranks[method][task] = rank_of(sc, pool, task)
        out["pools"][name] = {"entries": size, "ranks": dict(ranks)}
    return out


def report(res):
    for name, pool in res["pools"].items():
        print(f"Pool: {name} ({pool['entries']} entries)")
        for method, ranks in pool["ranks"].items():
            hits = "  ".join(f"top {c}: {sum(r <= c for r in ranks.values())}/{len(ranks)}" for c in CUTOFFS)
            print(f"  {method:7s} {hits}")
            print("          ranks " + " ".join(f"T{t:02d}:{r}" for t, r in sorted(ranks.items())))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--notes", default=paths.NOTES, help="directory with the knowledge notes (T<id>.md)")
    ap.add_argument("--embedding-model", default=os.environ.get("GISCLAW_EMBEDDING_MODEL"),
                    help="local path of BAAI/bge-large-en-v1.5; without it only BM25 is run")
    ap.add_argument("--json", help="also write the numbers to this file")
    args = ap.parse_args()
    if not args.embedding_model:
        print("No embedding model given: dense and hybrid retrieval are skipped.\n")
    res = compute(args.notes, args.embedding_model)
    report(res)
    stats.write_json(res, args.json)


if __name__ == "__main__":
    main()
