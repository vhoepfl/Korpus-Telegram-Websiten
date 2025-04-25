#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import heapq
import re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from gensim.models.fasttext import load_facebook_model
from tqdm import tqdm
import random


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine Similarity (−1 … 1)."""
    num = float(np.dot(v1, v2))
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return (num / den) if den else 0.0


def load_model(path: str):
    try:
        return load_facebook_model(path)
    except Exception as e:
        raise SystemExit(f"Konnte Modell '{path}' nicht laden: {e}")


def meets_criteria(
    word: str,
    base,
    tuned,
    min_base: int,
    min_tuned: int,
    pattern: re.Pattern | None,
):
    if base.wv.get_vecattr(word, "count") < min_base:
        return False
    if tuned.wv.get_vecattr(word, "count") < min_tuned:
        return False
    if pattern and not pattern.match(word):
        return False
    return True


def compute_shift(
    base,
    tuned,
    k: int,
    min_base: int,
    min_tuned: int,
    pattern: str | None,
):
    """Findet k größte Verschiebungen (cosine distance) im Embedding-Space durch Finetuning auf Telegram-Korpus."""
    regex = re.compile(pattern) if pattern else None
    common = set(base.wv.index_to_key) & set(tuned.wv.index_to_key)
    heap: List[Tuple[float, str]] = []

    for w in tqdm(common, unit="word"):
        if not meets_criteria(w, base, tuned, min_base, min_tuned, regex):
            continue
        d = 1 - cosine_sim(base.wv[w], tuned.wv[w]) # cosine distance to target
        if len(heap) < k:
            heapq.heappush(heap, (d, w))
        else:
            heapq.heappushpop(heap, (d, w))
    return sorted(heap, reverse=True)


def compute_towards_hybrid_local_global(
    base,
    tuned,
    targets: List[str],
    k: int,
    min_base: int,
    min_tuned: int,
    pattern: str | None,
    knn_w: int = 50,        # Nachbarn für Wort w 
    beta: float = 0.7,      # 0 = nur global, 1 = nur lokal
    eps: float = 1e-4,
    sample: int = 10_000,
) -> Dict[str, List[Tuple[float, str]]]:
    """
    Findet die k Wörter, die sich am stärksten einem Zielwort t annähern.
    sigma_global auf Basis vom globalen shift relativ zu Zielwort t, sigma_local auf Basis von w's k-NN.
    Hybrid-sigma = β·sigma_local + (1-β)·sigma_global
    """

    regex = re.compile(pattern) if pattern else None
    common = set(base.wv.index_to_key) & set(tuned.wv.index_to_key)

    # 1) Globale Standardabweichung des shift relativ zu Zielwort t
    sample_vocab = random.sample(common, min(sample, len(common)))
    sigma_global: Dict[str, float] = {}
    for t in targets:
        if t not in base.wv or t not in tuned.wv:
            print(f"⚠️ Target '{t}' fehlt - übersprungen.")
            continue
        tb, tt = base.wv[t], tuned.wv[t]
        
        deltas = [
            cosine_sim(base.wv[w], tb) - cosine_sim(tuned.wv[w], tt)
            for w in sample_vocab
        ]
        sigma_global[t] = max(np.std(deltas), eps)

    heaps: Dict[str, List[Tuple[float, str]]] = {t: [] for t in sigma_global}

    for w in tqdm(common, desc="Hybrid-Local-Global Z", unit="word"):
        if not meets_criteria(w, base, tuned, min_base, min_tuned, regex):
            continue
        bw, tw = base.wv[w], tuned.wv[w]

        for t, sigma_g in sigma_global.items():
            tb, tt = base.wv[t], tuned.wv[t]
            w_delta = cosine_sim(bw, tb) - cosine_sim(tw, tt)

            # Standardabweichung des shifts der KNN von w
            if beta == 0: 
                sigma_l = 0
            else: 
                knn_words_w = [n for n,_ in base.wv.most_similar(w, topn=knn_w) if n in tuned.wv]
                deltas_w = [
                    cosine_sim(base.wv[n], tb) - cosine_sim(tuned.wv[n], tt)
                    for n in knn_words_w
                ]
                sigma_l = max(np.std(deltas_w), eps)

            # Kombinieren zu hybrider Metrik
            sigma_h = beta * sigma_l + (1 - beta) * sigma_g
            z = w_delta / sigma_h

            heap = heaps[t]
            if len(heap) < k:
                heapq.heappush(heap, (z, w))
            else:
                heapq.heappushpop(heap, (z, w))

    return {t: sorted(h, reverse=True) for t, h in heaps.items()}


def write_csv(
    row_l: List[List[Tuple[float, str]]],
    path: Path,
    header: Tuple[str, str, str],
    moved_to: List[str] | None,
):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r_idx, rows in enumerate(row_l):
            for idx, row in enumerate(rows, 1):
                score, word = row[0], row[1]
                if moved_to:
                    writer.writerow([idx, word, f"{score:.6f}", moved_to[r_idx]])
                else:
                    writer.writerow([idx, word, f"{score:.6f}"])


def parse_args():
    ap = argparse.ArgumentParser(
    )
    ap.add_argument("base_model") 
    ap.add_argument("tuned_model")

    mode_grp = ap.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--mode",
        choices=["shift", "toward"],
        default="shift",
        help="shift = größte absolute Verschiebung (default); toward = Annäherung an Zielwort",
    )

    ap.add_argument("--targets", nargs="*", help="Zielwörter")
    ap.add_argument("--targets-file", type=Path, help="Datei mit Zielwörtern (eine Zeile = Wort)")
    ap.add_argument("--top", "-k", type=int, default=100)
    ap.add_argument("--min-count", nargs=2, type=int, metavar=("BASE", "TUNED"), default=(5, 5))
    ap.add_argument("--loc-glob-ratio", type=float, help="Verhältnis von lokaler und globaler Standardabweichung für Score-Berechnung. 0: Nur global, 1: Nur lokal", default=0.7)
    ap.add_argument("--pattern", help="Regex-Pattern - nur Wörter, die Pattern matchen, in Betracht gezogen")
    ap.add_argument("--out", "-o", type=Path, help="Pfad für Output-Datei")
    return ap.parse_args()


def main():
    args = parse_args()

    base_min, tuned_min = args.min_count
    base = load_model(args.base_model)
    tuned = load_model(args.tuned_model)

    if base.wv.vector_size != tuned.wv.vector_size:
        raise SystemExit(f"Modelldimensionen in base model {base.wv.vector_size} und finetuned model {tuned.wv.vector_size} unterscheiden sich.")

    rows_per_target: List[List[Tuple[float, str]]] = []
    moved_to: List[str] | None = None

    # SHIFT
    if args.mode == "shift":
        shift_rows = compute_shift(
            base, tuned, args.top, base_min, tuned_min, args.pattern
        )
        rows_per_target.append(shift_rows)
        header = ("rank", "word", "cosine_distance")

    # TOWARDS
    else:
        targets: List[str] = []
        if args.targets:
            targets.extend(args.targets)
        if args.targets_file:
            targets.extend(
                l.strip()
                for l in args.targets_file.read_text(encoding="utf-8").splitlines()
                if l.strip()
            )
        if not targets:
            raise SystemExit("--mode toward braucht Zielwörter via --targets oder --targets-file")

        moved_to = targets
        header = ("rank", "word", "score", "target")

        heap_dict = compute_towards_hybrid_local_global(
            base,
            tuned,
            targets = targets,
            k = args.top,
            min_base = base_min,
            min_tuned = tuned_min,
            pattern = args.pattern,
            beta=args.loc_glob_ratio
        )

        for t in targets:
            rows_per_target.append(heap_dict.get(t, []))

    if args.out:
        write_csv(rows_per_target, args.out, header, moved_to)
        print(f"✅ CSV gespeichert: {args.out}")
    else:
        if args.mode == "shift":
            print("\nRank  Wort                 Score")
            for idx, (score, w) in enumerate(rows_per_target[0], 1):
                print(f"{idx:>4} {w:<20} {score:.4f}")
        else:  # toward
            for target, rows in zip(moved_to, rows_per_target):
                print(f"\nTarget: {target}")
                print("Rank  Wort                 ΔScore")
                for idx, (score, w) in enumerate(rows, 1):
                    print(f"{idx:>4} {w:<20} {score:.4f}")


if __name__ == "__main__":
    main()
