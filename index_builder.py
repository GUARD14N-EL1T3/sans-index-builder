#!/usr/bin/env python3

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import pymupdf
import pytesseract
from PIL import Image
from PIL.ImageFile import ImageFile
from tqdm import tqdm
from wordfreq import zipf_frequency

DEBUG = True

# -----------------------------------------------------------------------------
# Default argument option values
# -----------------------------------------------------------------------------


DEFAULT_DIR_BASE = "."
DEFAULT_DIR_CACHE = "cache"
DEFAULT_DIR_WORDLIST = "wordlists"
DEFAULT_FILE_EXCLUDE = "exclude.txt"
DEFAULT_FILE_INCLUDE = "include.txt"
DEFAULT_FILE_OUTPUT = "index.csv"
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_CONFIDENCE = 40
DEFAULT_OCR_MARGIN_BOTTOM = 5
DEFAULT_OCR_MARGIN_TOP = 0
DEFAULT_OCR_SKIP_PAGES = 2
DEFAULT_STRIP_MAX_WORDS = 20
DEFAULT_STRIP_MIN_WORDS = 3
DEFAULT_STRIP_THRESHOLD = 50
DEFAULT_TERM_OCCURRENCES = 2
DEFAULT_TERM_ZIPF = 3.6


# -----------------------------------------------------------------------------
# Regex
# -----------------------------------------------------------------------------


RE_ACR = re.compile(r"\b[A-Z]{2,6}\b")
RE_TERM_ACR = re.compile(r"^(.*\S)\s\(([A-Za-z0-9]{2,8})\)$")
RE_TERM_ACR_FIRST = re.compile(r"\b([A-Z]{2,8})\s\((?!e\.g\.|i\.e\.)((?:[A-Z][a-zA-Z0-9\-]*\s){1,5}[A-Z][a-zA-Z0-9\-]*)\)")
RE_TERM_ACR_LAST = re.compile(r"\b((?:[A-Z][a-zA-Z0-9\-]*\s){1,5}[A-Z][a-zA-Z0-9\-]*)\s\(([A-Z]{2,8})\)")
RE_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|[.,;:!?()]")


# -----------------------------------------------------------------------------
# Wordlist
# -----------------------------------------------------------------------------


DEFAULT_CONNECTORS = {
    "&",
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "vs.",
    "vs",
    "with",
}

DEFAULT_FORBIDDEN_STARTERS = {
    "a",
    "additionally",
    "agenda",
    "all",
    "also",
    "although",
    "an",
    "any",
    "are",
    "as",
    "at",
    "because",
    "by",
    "can",
    "chapter",
    "could",
    "course",
    "did",
    "do",
    "does",
    "each",
    "ensure",
    "every",
    "exercise",
    "figure",
    "finally",
    "first",
    "for",
    "from",
    "furthermore",
    "had",
    "has",
    "have",
    "he",
    "here",
    "however",
    "i",
    "if",
    "in",
    "introduction",
    "is",
    "it",
    "its",
    "lab",
    "let",
    "lets",
    "make",
    "may",
    "might",
    "module",
    "moreover",
    "must",
    "no",
    "note",
    "now",
    "objectives",
    "on",
    "our",
    "overview",
    "page",
    "please",
    "refer",
    "review",
    "second",
    "section",
    "see",
    "she",
    "should",
    "since",
    "slide",
    "so",
    "some",
    "summary",
    "table",
    "that",
    "the",
    "their",
    "then",
    "there",
    "therefore",
    "these",
    "they",
    "this",
    "those",
    "thus",
    "to",
    "use",
    "using",
    "was",
    "we",
    "were",
    "when",
    "while",
    "with",
    "would",
    "you",
    "your",
}

DEFAULT_STOPWORDS = {
    "ALL",
    "AND",
    "ARE",
    "BOY",
    "BUT",
    "CAN",
    "DAY",
    "DID",
    "FOR",
    "GET",
    "GIAC",
    "HAS",
    "HER",
    "HIM",
    "HIS",
    "HOW",
    "ITS",
    "LET",
    "MAN",
    "NEW",
    "NOT",
    "NOW",
    "OLD",
    "ONE",
    "OUR",
    "OUT",
    "PUT",
    "SANS",
    "SAY",
    "SEE",
    "SHE",
    "THE",
    "TOO",
    "TWO",
    "USE",
    "WAS",
    "WAY",
    "WHO",
    "YOU",
}

WORDLIST_FILES = {
    "connectors": (
        "connectors.txt",
        DEFAULT_CONNECTORS,
        False,
    ),
    "forbidden_starters": (
        "forbidden_starters.txt",
        DEFAULT_FORBIDDEN_STARTERS,
        False,
    ),
    "stopwords": (
        "stopwords.txt",
        DEFAULT_STOPWORDS,
        True,
    ),
}


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------


def log_msg(name: str, msg: str) -> None:
    print(f"[{name}] {msg}", file=sys.stderr)


def iterate_pages(iterable: Iterable[Any], desc: str | None = None) -> enumerate:
    return enumerate(tqdm(iterable=iterable, desc=desc, unit="page", file=sys.stderr), start=1)


def search_term_variants(term: str) -> str | tuple[str, str]:
    matches = RE_TERM_ACR.match(term.strip())
    if matches:
        full = matches.group(1)
        acr = matches.group(2)
        return (full.strip(), acr.strip())
    return term.strip()


# -----------------------------------------------------------------------------
# Term list building
# -----------------------------------------------------------------------------


def read_terms_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def load_term_set(path: str) -> set[str]:
    if not path:
        return None

    term_list = read_terms_file(path)

    term_set = set[str | tuple[str, str]]()
    for term in term_list:
        term_set.add(search_term_variants(term))

    return term_set


def load_wordlists(path: str) -> defaultdict[str, list[str]]:
    words = defaultdict[str, list[str]](list[str])
    for key, (filename, default, uppercase) in WORDLIST_FILES.items():
        path = os.path.join(path, filename)
        if os.path.exists(path):
            words[key] = read_terms_file(path)
            if uppercase:
                words[key] = [w.upper() for w in words[key]]
            else:
                words[key] = [w.lower() for w in words[key]]
            log_msg("load_wordlists", f"Loaded {len(words[key])} entries from '{path}'")
        else:
            words[key] = default
            log_msg("load_wordlists", f"Path '{path}' does not exist. Using default wordlist")
    return words


# -----------------------------------------------------------------------------
# OCR extraction and caching
# -----------------------------------------------------------------------------


def get_ocr_cache(pdf_path: str, dir: str, confidence: int, dpi: int) -> str:
    if not dir:
        return None

    file_name = os.path.basename(pdf_path)
    cache_name = os.path.splitext(file_name)
    return str(os.path.join(dir, f"{cache_name[0]}-{confidence}-{dpi}.json"))


def load_ocr_cache(path: str) -> list[tuple[int, Any]]:
    if not path or not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(int(p), t) for p, t in raw]


def save_ocr_cache(path: str, data: Any) -> None:
    if not path:
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def get_crop_rect(page: pymupdf.Page, margin_bottom: float, margin_top: float) -> pymupdf.Rect:
    rect: pymupdf.Rect = page.rect
    top = rect.y0 + (rect.height * margin_top)
    bottom = rect.y1 - (rect.height * margin_bottom)
    return pymupdf.Rect(rect.x0, top, rect.x1, bottom)


def render_pdf_image(path: str, dpi: int, margin_bottom: int = 0, margin_top: int = 0, password: str = "", skip: int = 0) -> list[ImageFile]:
    doc = pymupdf.Document(path)
    if doc.needs_pass and (not password or not doc.authenticate(password)):
        raise ValueError("PDF is password-protected and the supplied password did not work or not provided.")

    images = list[ImageFile]()
    for i, page in iterate_pages(doc, f"Rendering pages at {dpi} DPI"):
        if i < skip:
            continue

        crop_rect = get_crop_rect(page, margin_bottom / 100, margin_top / 100)
        pix = page.get_pixmap(dpi=dpi, clip=crop_rect)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)

    return images


def ocr_images(images: list[ImageFile], confidence: float) -> list[tuple[int, str]]:
    page_blobs = list[tuple[int, str]]()
    for i, img in iterate_pages(images, "OCR"):
        data = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)
        word_blob = list[str]()
        for word, ocr_confidence in zip(data["text"], data["conf"]):
            word = str(word).strip()
            if not word or ocr_confidence < confidence:
                continue

            word_blob.append(word)

        page_blobs.append((i, " ".join(word_blob)))

    return page_blobs


# -----------------------------------------------------------------------------
# Header/footer stripping
# -----------------------------------------------------------------------------


def strip_repeating_lines(page_blobs: list[tuple[int, str]], max_len: int, min_len: int, threshold: float) -> list[tuple[int, list[str]]]:
    tokenized = [(page_num, text.split()) for page_num, text in page_blobs]
    cutoff = max(2, int(len(page_blobs) * threshold))

    def ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
        return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    repeating_by_len = {}
    for i in range(max_len, min_len - 1, -1):
        counts = Counter()
        for _, words in tokenized:
            counts.update(set(ngrams(words, i)))  # count once per page
        repeating_by_len[i] = {g for g, c in counts.items() if c >= cutoff}

    result = list[tuple[int, list[str]]]()
    repeating_display = set()
    for page_num, words in tokenized:
        kept = list[str]()
        i = 0
        while i < len(words):
            matched_len = None
            for n in range(max_len, min_len - 1, -1):
                if i + n <= len(words) and tuple(words[i : i + n]) in repeating_by_len[n]:
                    matched_len = n
                    break
            if matched_len:
                repeating_display.add(" ".join(words[i : i + matched_len]))
                i += matched_len
            else:
                kept.append(words[i])
                i += 1
        result.append((page_num, [" ".join(kept)]))

    log_msg("strip_repeating_lines", f"Stripped {len(repeating_display)} repeating header/footer word-sequence(s)")

    return result


# -----------------------------------------------------------------------------
# Term matching / extraction
# -----------------------------------------------------------------------------


def find_acronym_definitions(page_blobs: list[tuple[int, str]]) -> defaultdict[str, str]:
    votes = defaultdict[str, Counter](Counter)
    for page_num, lines in page_blobs:
        text = "\n".join(lines)
        for matches in RE_TERM_ACR_LAST.finditer(text):
            full, acr = matches.group(1).strip(), matches.group(2).strip()
            votes[acr][full] += 1
        for matches in RE_TERM_ACR_FIRST.finditer(text):
            acr, full = matches.group(1).strip(), matches.group(2).strip()
            votes[acr][full] += 1
    return {acr: counter.most_common(1)[0][0] for acr, counter in votes.items()}


def build_candidate_phrases(text: str, connectors: list[str]) -> str:
    phrase = list[str]()
    connector_buf = list[str]()

    def flush():
        if phrase:
            yield " ".join(phrase)
        phrase.clear()
        connector_buf.clear()

    for token in RE_TOKEN.findall(text):
        if token in ".,;:!?()":
            yield from flush()
            continue
        token_lower = token.lower()
        if token[0].isupper():
            if connector_buf:
                phrase.extend(connector_buf)
                connector_buf.clear()
            phrase.append(token)
        elif token_lower in connectors and phrase:
            connector_buf.append(token)
        else:
            yield from flush()
    yield from flush()


def is_forbidden_start(phrase: str, forbidden_starters: list[str]) -> bool:
    first_word = phrase.split()[0].lower().strip(".,;:!?()")
    return first_word in forbidden_starters


def is_common_english_word(term: str, zipf: float) -> bool:
    if " " in term or "-" in term or any(c.isdigit() for c in term):
        return False
    return zipf_frequency(term.lower(), "en") > zipf


def extract_terms(
    page_blobs: list[tuple[int, str]],
    wordlists: defaultdict[str, list[str]],
    zipf: float,
    exclude_set: set[str] | None = None,
) -> defaultdict[str, set[int]]:
    exclude_set = exclude_set or set()
    connectors = wordlists["connectors"]
    forbidden_starters = wordlists["forbidden_starters"]
    stopwords = wordlists["stopwords"]

    acronym_map = find_acronym_definitions(page_blobs)
    full_to_acr = {full.lower(): acr for acr, full in acronym_map.items()}

    term_index = defaultdict[str, set[int]](set)
    for page_num, lines in page_blobs:
        text = "\n".join(lines)

        for matches in RE_ACR.finditer(text):
            acr = matches.group(0)
            if acr in stopwords or acr.lower() in exclude_set:
                continue
            term = f"{acronym_map[acr]} ({acr})" if acr in acronym_map else acr
            term_index[term].add(page_num)

        for phrase in build_candidate_phrases(text, connectors):
            if phrase.lower() in exclude_set:
                continue
            if len(phrase) < 2 or phrase.upper() in stopwords:
                continue
            if is_common_english_word(phrase, zipf):
                continue
            if is_forbidden_start(phrase, forbidden_starters):
                continue
            acr = full_to_acr.get(phrase.lower())
            term = f"{acronym_map[acr]} ({acr})" if acr else phrase
            term_index[term].add(page_num)

    return term_index


def build_index(term_index: defaultdict[str, set[str]], book: int, min_occurrences: int) -> list[tuple[str, int, int]]:
    rows = list[tuple[str, int, int]]()
    for term, page_nums in term_index.items():
        if len(page_nums) >= min_occurrences:
            index: str = " ".join(str(p) for p in sorted(page_nums))
            rows.append((term, book, index))

    log_msg("build_index", f"Built index with {len(rows)} terms")
    return rows


# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "path",
        help="Path to target PDF",
    )
    ap.add_argument(
        "book",
        # default=1,
        help="Book number in course book series (e.g. SEC565.1 is book number 1, and SEC565.4 is book number 4)",
        type=int,
    )
    ap.add_argument(
        "-b",
        "--base",
        default=DEFAULT_DIR_BASE,
        dest="base",
        help="Path to base directory. All relative paths use this base path",
    )
    ap.add_argument(
        "-c",
        "--cache",
        default=DEFAULT_DIR_CACHE,
        dest="cache",
        help="Path to cache directory",
    )
    ap.add_argument(
        "--confidence",
        default=DEFAULT_OCR_CONFIDENCE,
        dest="confidence",
        help="Drop OCR words below this confidence 0-100. Adjust to not detect watermarks or if your output is jumbled.",
        type=float,
    )
    ap.add_argument(
        "-d",
        "--dpi",
        default=DEFAULT_OCR_DPI,
        dest="dpi",
        help="Render DPI for OCR",
        type=int,
    )
    ap.add_argument(
        "-e",
        "--exclude-terms",
        dest="exclude",
        help="Path to a file of terms/phrases to always exclude (one term / line), "
        "include generic boilerplate or other garbage not needed ever."
        "Applied in both 'build' and 'list' modes.",
    )
    ap.add_argument(
        "-i",
        "--include-terms",
        dest="include",
        help="Path to a file of terms/phrases to always exclude (one term / line), "
        "include generic boilerplate or other garbage not needed ever."
        "Applied in both 'build' and 'list' modes.",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        dest="no_cache",
        help="Ignore cache option flags and force OCR text extraction",
    )
    ap.add_argument(
        "--margin-bottom",
        default=DEFAULT_OCR_MARGIN_BOTTOM,
        dest="margin_bottom",
        help="Fraction of page height to crop from the bottom before OCR "
        "(e.g. '8' crops the bottom 8%%). Excludes footer/watermark text "
        "by position instead of by content, so it works even when OCR "
        "reads that text differently from page to page.",
        type=float,
    )
    ap.add_argument(
        "--margin-top",
        default=DEFAULT_OCR_MARGIN_TOP,
        dest="margin_top",
        help="Fraction of page height to crop from the top before OCR (e.g. '8' crops the top 8%%).",
        type=float,
    )
    ap.add_argument(
        "--max-words",
        default=DEFAULT_STRIP_MAX_WORDS,
        dest="max_words",
        help="Maximum length of a repeating word-sequence to strip as header/footer",
        type=int,
    )
    ap.add_argument(
        "--min-words",
        default=DEFAULT_STRIP_MIN_WORDS,
        dest="min_words",
        help="Minimum length of a repeating word-sequence to strip as header/footer",
        type=int,
    )
    ap.add_argument(
        "-n",
        "--occurrences",
        default=DEFAULT_TERM_OCCURRENCES,
        dest="occurrences",
        help="Minimum distinct pages on which a term must appear",
        type=int,
    )
    ap.add_argument(
        "-o",
        "--output",
        default=DEFAULT_FILE_OUTPUT,
        dest="output",
        help="Output path for CSV file containing index",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        dest="overwrite",
        help="Overwrite a potentially existing output file path",
    )
    ap.add_argument(
        "-p",
        "--password",
        dest="password",
        help="Encrypted PDF password and can be set using the environment variable 'PDF_PASSWORD'",
    )
    ap.add_argument(
        "--threshold",
        default=DEFAULT_STRIP_THRESHOLD,
        dest="threshold",
        help="Ratio of pages that will trigger stripping a word-sequence",
        type=float,
    )
    ap.add_argument(
        "-w",
        "--wordlist",
        default=DEFAULT_DIR_WORDLIST,
        dest="wordlist",
        help="Directory containing wordlist text files (one term / line). connectors.txt / forbidden_starters.txt / stopwords.txt",
    )
    ap.add_argument(
        "-z",
        "--zipf",
        default=DEFAULT_TERM_ZIPF,
        dest="zipf",
        help="Drops single common-English words scoring above the value on the Zipf frequency scale. "
        "Requires 'pip install wordfreq'; Does not take effect if not installed. "
        "Lower = stricter (drops more borderline words), higher = more permissive.",
        type=float,
    )
    args = ap.parse_args()

    if args.no_cache:
        args.cache = None
        log_msg("main", "Forcing no cache")

    if not args.password:
        args.password = os.environ.get("PDF_PASSWORD")
        if not args.password:
            log_msg("main", "No password provided")

    cache_path = get_ocr_cache(args.path, args.cache, args.confidence, args.dpi)
    wordlists = load_wordlists(args.wordlist)
    exclude = load_term_set(args.exclude)
    include = load_term_set(args.include)
    page_blobs = load_ocr_cache(cache_path)

    try:
        if page_blobs == None:
            images = render_pdf_image(
                args.path,
                args.dpi,
                margin_bottom=args.margin_bottom,
                margin_top=args.margin_top,
                password=args.password,
            )
            page_blobs = ocr_images(images, args.confidence)
    except (OSError, ValueError) as e:
        log_msg("main", f"ERROR: {e}")
        sys.exit(1)

    if args.cache:
        if not os.path.exists(args.cache):
            os.mkdir(args.cache)
            log_msg("main", f"Created '{args.cache}' directory")
        save_ocr_cache(cache_path, page_blobs)
        log_msg("main", f"Cached OCR text to '{cache_path}'")
    else:
        log_msg("main", "Skipping saving to cache")

    page_blobs = strip_repeating_lines(
        page_blobs=page_blobs,
        max_len=args.max_words,
        min_len=args.min_words,
        threshold=args.threshold,
    )

    terms_index = extract_terms(
        page_blobs=page_blobs,
        wordlists=wordlists,
        zipf=args.zipf,
        exclude_set=exclude,
        # include_set=include,
    )

    rows = build_index(terms_index, args.book, args.occurrences)

    if not args.overwrite and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            if f.readline() != "Term,Book,Page\n":
                log_msg("main", "WARNING: Row header is not what is expected")

    open_mode = "w" if args.overwrite else "a"
    with open(args.output, open_mode, encoding="utf-8") as f:
        writer = csv.writer(f)
        if args.overwrite:
            writer.writerow(["Term", "Book", "Page"])
        writer.writerows(rows)
    action = "Wrote" if args.overwrite else "Appended"
    log_msg("main", f"{action} to '{args.output}'")


if __name__ == "__main__":
    main()
