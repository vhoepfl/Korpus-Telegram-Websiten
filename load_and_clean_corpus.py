from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List
merged_phrases = set()
import regex

def cleanup(text: str, re_cleanup: bool, lowercase: bool, include_emojis: bool) -> str:
    if lowercase:
        text = text.lower()
    if re_cleanup:
        # Remove link placeholders, no value for text
        text = re.sub(r"\[link\]", " ", text)

        # Replace punctuation (except normal dot) with whitespace, keep words+digits
        if include_emojis: 
            emoji = regex.compile("([\p{Extended_Pictographic}\p{Emoji_Modifier_Base}\p{Emoji_Modifier}\p{Emoji_Component}]+)")
            text = emoji.sub(r" \1 ", text)
            non_emoji = regex.compile("[^\p{Extended_Pictographic}\p{Emoji_Modifier_Base}\p{Emoji_Modifier}\p{Emoji_Component}\p{Letter}\p{Decimal_Number}#]+")
            text = non_emoji.sub(" ", text)
        else: 
            text = re.sub(r"[^\w\n\d#]+", " ", text)
        # delete leading/trailing spaces per line
        text = re.sub(r"^\s+|\s+(?=\n)", "", text, flags=re.MULTILINE)

    # collapse multi‑spaces (but not newlines) to single space
    text = re.sub(r"[^\S]+", " ", text)
    return text


def extract_text_from_xml(file_path: str | os.PathLike, do_cleanup:bool, include_emojis:bool) -> List[str]:
    """Parst <body><text>…</text></body> XML und säubert Zeilen."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        if root.tag != "body":
            return []
        elems = root.findall(".//text")

        print(f" {file_path} erfolgreich gelesen, enthält {len([1 for e in elems if e.text])} einzelne Texte")
        if do_cleanup:
            return [cleanup(e.text, re_cleanup=True, lowercase=True, include_emojis=include_emojis) for e in elems if e.text]
        else: 
            return [cleanup(e.text, re_cleanup=False, lowercase=False, include_emojis=include_emojis) for e in elems if e.text]

    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {file_path} konnte nicht gelesen werden: {exc}")
        return []


def iter_xml_files(folder: str | os.PathLike) -> Iterable[str]:
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.endswith(".xml"):
                yield os.path.join(root, fn)



def bigram_stats(sentences: Iterable[List[str]]) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    """Zähle Unigramme + Bigrams über alle Sätze."""
    uni = Counter()
    bi = Counter()
    for sent in sentences:
        if not sent:
            continue
        uni.update(sent)
        # bigrams (adjacent only – wie im Paper)
        bi.update(zip(sent, sent[1:]))
    return uni, bi


def score_bigrams(
    uni: Counter[str],
    bi: Counter[tuple[str, str]],
    delta: float,
) -> dict[tuple[str, str], float]:
    """Compute Mikolov score for each bigram."""
    scores = {}
    for (w1, w2), c12 in bi.items():
        score = (c12 - delta) / (uni[w1] * uni[w2])
        scores[(w1, w2)] = score
    return scores


def merge_phrases_once(
    sentences: List[List[str]],
    scores: dict[tuple[str, str], float],
    threshold: float,
    join_str: str = "_",
) -> List[List[str]]:
    """Ersetze Bigramme deren score>threshold durch zusammengefügte Tokens."""
    out_sents: list[list[str]] = []
    for sent in sentences:
        i = 0
        merged: list[str] = []
        while i < len(sent):
            if i < len(sent) - 1 and scores.get((sent[i], sent[i + 1]), 0.0) > threshold:
                merged_phrases.add(f"{sent[i]}{join_str}{sent[i+1]}") # Save all merged phrases
                merged.append(f"{sent[i]}{join_str}{sent[i+1]}")
                i += 2  # skip next token
            else:
                merged.append(sent[i])
                i += 1
        out_sents.append(merged)
    return out_sents


def phrase_mining(
    raw_sentences: List[str],
    passes: int = 2,
    delta: float = 5.0,
    thresholds: list[float] | None = None,
    join_str: str = "_",
) -> List[List[str]]:
    """Mehrstufige Phrasenerkennung."""
    if thresholds is None:
        thresholds = [1e-5, 1e-6, 1e-7][:passes]
    sentences = [sent.split() for sent in raw_sentences]
    for p in range(passes):
        uni, bi = bigram_stats(sentences)
        scores = score_bigrams(uni, bi, delta)
        sentences = merge_phrases_once(sentences, scores, thresholds[p], join_str)
        print(f"[PhrasePass {p+1}] merged → {sum(len(s) for s in sentences)} tokens")
    return sentences

############################################################
# 3. Saving Helpers
############################################################

def save_corpus(sentences: List[List[str]], outfile: str | os.PathLike):
    with open(outfile, "w", encoding="utf-8") as fh:
        for tokens in sentences:
            fh.write(" ".join(tokens) + "\n")
    print(f"[OK] Korpus @ {outfile}")

############################################################
# 4. Command‑Line Entry‑Point
############################################################

def save_merged(outfile):
    with open(outfile, "w", encoding="utf-8") as fw:
        for phrase in sorted(merged_phrases):
            fw.write(phrase + "\n")
    print(f"[OK] Merged Vocabulary @ {outfile}")

def main():
    ap = argparse.ArgumentParser(description="Build phrased corpus from XML folder.")
    ap.add_argument("folder", help="Path to folder with XML files")
    ap.add_argument("--out_corpus", default="corpus_cleaned.txt")
    ap.add_argument("--do_cleanup", type=str, default="yes")
    ap.add_argument("--include_emojis", type=str, default="yes")
    ap.add_argument("--passes", type=int, default=2, help="# of phrase passes (2‑4 sinnvoll)")
    ap.add_argument("--delta", type=float, default=5.0, help="Discount coefficient δ")
    ap.add_argument("--thresholds", nargs="*", type=float, help="Phrase threshold per pass")
    args = ap.parse_args()

    if args.do_cleanup.lower() == "yes": 
        do_cleanup = True
    elif args.do_cleanup.lower() == "no": 
        do_cleanup = False
    else: 
        print("Please enter either 'yes' or 'no' for --do_cleanup")

    if args.include_emojis.lower() == "yes": 
        include_emojis = True
    elif args.include_emojis.lower() == "no": 
        include_emojis = False
    else: 
        print("Please enter either 'yes' or 'no' for --include_emojis")


    raw_texts: list[str] = []
    for fp in iter_xml_files(args.folder):
        raw_texts.extend(extract_text_from_xml(fp, do_cleanup=do_cleanup, include_emojis=include_emojis))
    print(f"[INFO] {len(raw_texts)} Text‑Blöcke geladen")

    print("do cleanup of text: ", args.do_cleanup)
    if do_cleanup:
        # 2) phrase mining
        sentences = phrase_mining(
            raw_texts,
            passes=args.passes,
            delta=args.delta,
            thresholds=args.thresholds,
        )
        # save all merged terms
        save_merged('merged_vocab_' + args.out_corpus)
    else: 
        sentences = [sent.split() for sent in raw_texts]

    save_corpus(sentences, args.out_corpus)


if __name__ == "__main__":
    main()
