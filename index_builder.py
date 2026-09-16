#!/usr/bin/env python3

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pymupdf
import pytesseract
from PIL import Image
from tqdm import tqdm
from wordfreq import zipf_frequency

# -----------------------------------------------------------------------------
# Default argument option values
# -----------------------------------------------------------------------------


DEFAULT_CACHE = "cache"
DEFAULT_DPI = 300
DEFAULT_OCCURRENCES = 2
DEFAULT_OCR_CONFIDENCE = 40
DEFAULT_OUTPUT = "index.csv"
DEFAULT_STRIP_MAX_LEN = 10
DEFAULT_STRIP_MIN_LEN = 3
DEFAULT_STRIP_THRESHOLD = 0.5
DEFAULT_WORDLIST = "wordlists"
DEFAULT_ZIPF = 3.6


# -----------------------------------------------------------------------------
# Regex
# -----------------------------------------------------------------------------


RE_ACR = re.compile(r"\b[A-Z]{2,6}\b")

RE_TERM_ACR = re.compile(r"^(.*\S)\s\(([A-Za-z0-9]{2,8})\)$")

RE_TERM_ACR_FIRST = re.compile(
    r"\b([A-Z]{2,8})\s\((?!e\.g\.|i\.e\.)((?:[A-Z][a-zA-Z0-9\-]*\s){1,5}[A-Z][a-zA-Z0-9\-]*)\)"
)

RE_TERM_ACR_LAST = re.compile(
    r"\b((?:[A-Z][a-zA-Z0-9\-]*\s){1,5}[A-Z][a-zA-Z0-9\-]*)\s\(([A-Z]{2,8})\)"
)

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


def log_msg(name, msg):
    print(f"[{name}] {msg}", file=sys.stderr)


def iterate_pages(iterable, desc):
    return enumerate(
        tqdm(
            iterable,
            desc=desc,
            unit="page",
            file=sys.stderr,
        ),
        start=1,
    )


def search_term_variants(term):
    matches = RE_TERM_ACR.match(term.strip())
    if matches:
        full, acr = matches.group(1).strip(), matches.group(2).strip()
        return [full, acr]
    return [term.strip()]


def build_index(term_index, book, min_occurrences=2):
    rows = []
    for term, page_nums in term_index.items():
        if len(page_nums) >= min_occurrences:
            index = " ".join(str(p) for p in sorted(page_nums))
            rows.append((term, book, index))

    log_msg("build_index", f"Built index with {len(rows)} terms")
    return rows


# -----------------------------------------------------------------------------
# Term list building
# -----------------------------------------------------------------------------


def read_terms_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def load_exclude_set(path):
    excluded = read_terms_file(path)

    variants = set()
    for term in excluded:
        for variant in search_term_variants(term):
            variants.add(variant.lower())
    excluded += variants

    log_msg(
        "load_exclude_set",
        f"Loaded {len(excluded)} excluded term(s) from '{path}'",
    )
    return excluded


def load_wordlists(wordlist):
    words = defaultdict(list)
    for key, (filename, default, uppercase) in WORDLIST_FILES.items():
        path = os.path.join(wordlist, filename)
        if os.path.exists(path):
            words[key] = read_terms_file(path)
            if uppercase:
                words[key] = [w.upper() for w in words[key]]
            else:
                words[key] = [w.lower() for w in words[key]]
            log_msg(
                "load_wordlists",
                f"Loaded {len(words[key])} entries from '{path}'",
            )
        else:
            words[key] = default
            log_msg(
                "load_wordlists",
                f"Path '{path}' does not exist. Using default wordlist",
            )
    return words


# -----------------------------------------------------------------------------
# OCR extraction and caching
# -----------------------------------------------------------------------------


def get_ocr_cache(path, dir, confidence, dpi):
    file_name = os.path.basename(path)
    cache_name = os.path.splitext(file_name)
    if not cache_name[1]:
        log_msg(
            "get_ocr_cache",
            "WARNING: PDF path has no extension! This will cause another file "
            "with the same name with an extension to use the same cached file",
        )
    return str(os.path.join(dir, f"{cache_name[0]}-{confidence}-{dpi}.json"))


def load_ocr_cache(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(int(p), t) for p, t in raw]


def save_ocr_cache(path, pages):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pages, f)


def render_pages(path, dpi, password=None):
    doc = pymupdf.open(path)
    if doc.needs_pass and (not password or not doc.authenticate(password)):
        raise ValueError(
            "PDF is password-protected and the supplied password did not work or not provided."
        )

    images = []
    for i, page in iterate_pages(doc, "Rendering pages"):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)

    doc.close()
    return images


def ocr_pages(path, confidence, dpi, password=None):
    log_msg(
        "ocr_pages",
        f"Rendering pages at {dpi} DPI ...",
    )

    images = render_pages(path, dpi, password)
    log_msg(
        "ocr_pages",
        f"{len(images)} pages rendered. Running OCR with {confidence}% confidence (time-intensive) ...",
    )

    pages = []
    for i, img in iterate_pages(images, "OCR"):
        data = pytesseract.image_to_data(
            img,
            lang="eng",
            output_type=pytesseract.Output.DICT,
        )
        words = []
        for word, ocr_conf in zip(data["text"], data["conf"]):
            word = word.strip()
            if not word:
                continue

            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = -1
            if ocr_conf < confidence:
                continue

            words.append(word)

        pages.append((i, " ".join(words)))

    return pages


def get_pages(path, cache_dir, confidence, dpi, password=None):
    if cache_dir and os.path.exists(cache_dir):
        cache_path = get_ocr_cache(path, cache_dir, confidence, dpi)
        if os.path.exists(cache_path):
            log_msg("get_pages", f"Loading cached OCR text from '{cache_path}' ...")
            return load_ocr_cache(cache_path)
    else:
        log_msg("get_pages", "No cache found or wanted")

    pages = ocr_pages(
        path,
        confidence=confidence,
        dpi=dpi,
        password=password,
    )

    if cache_dir:
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
            log_msg("get_pages", f"Created '{cache_dir}' directory")
        cache_path = get_ocr_cache(path, cache_dir, confidence, dpi)
        save_ocr_cache(cache_path, pages)
        log_msg("get_pages", f"Cached OCR text to '{cache_path}'")
    else:
        log_msg("get_pages", "Skipping saving to cache")

    return pages


# -----------------------------------------------------------------------------
# Header/footer stripping
# -----------------------------------------------------------------------------


def strip_repeating_lines(pages, max_len, min_len, threshold):
    tokenized = [(page_num, text.split()) for page_num, text in pages]
    cutoff = max(2, int(len(pages) * threshold))

    def ngrams(words, n):
        return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    repeating_by_len = {}
    for i in range(max_len, min_len - 1, -1):
        counts = Counter()
        for _, words in tokenized:
            counts.update(set(ngrams(words, i)))  # count once per page
        repeating_by_len[i] = {g for g, c in counts.items() if c >= cutoff}

    result = []
    repeating_display = set()
    for page_num, words in tokenized:
        kept = []
        i = 0
        while i < len(words):
            matched_len = None
            for n in range(max_len, min_len - 1, -1):
                if (
                    i + n <= len(words)
                    and tuple(words[i : i + n]) in repeating_by_len[n]
                ):
                    matched_len = n
                    break
            if matched_len:
                repeating_display.add(" ".join(words[i : i + matched_len]))
                i += matched_len
            else:
                kept.append(words[i])
                i += 1
        result.append((page_num, [" ".join(kept)]))

    log_msg(
        "strip_repeating_lines",
        f"Stripped {len(repeating_display)} repeating header/footer word-sequence(s)",
    )

    return result


# -----------------------------------------------------------------------------
# Term matching / extraction
# -----------------------------------------------------------------------------


def find_acronym_definitions(pages):
    votes = defaultdict(Counter)
    for page_num, lines in pages:
        text = "\n".join(lines)
        for matches in RE_TERM_ACR_LAST.finditer(text):
            full, acr = matches.group(1).strip(), matches.group(2)
            votes[acr][full] += 1
        for matches in RE_TERM_ACR_FIRST.finditer(text):
            acr, full = matches.group(1), matches.group(2).strip()
            votes[acr][full] += 1
    return {acr: counter.most_common(1)[0][0] for acr, counter in votes.items()}


def build_candidate_phrases(text, connectors):
    phrase = []
    connector_buf = []

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


def is_forbidden_start(phrase, forbidden_starters):
    first_word = phrase.split()[0].lower().strip(".,;:!?()")
    return first_word in forbidden_starters


def is_common_english_word(term, max_zipf):
    if " " in term or "-" in term or any(c.isdigit() for c in term):
        return False
    return zipf_frequency(term.lower(), "en") > max_zipf


def extract_terms(pages, wordlists, zipf, exclude_set=None):
    exclude_set = exclude_set or set()
    connectors = wordlists["connectors"]
    forbidden_starters = wordlists["forbidden_starters"]
    stopwords = wordlists["stopwords"]

    acronym_map = find_acronym_definitions(pages)
    full_to_acr = {full.lower(): acr for acr, full in acronym_map.items()}

    term_index = defaultdict(set)
    for page_num, lines in pages:
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
            if is_forbidden_start(phrase, forbidden_starters):
                continue
            if len(phrase) < 2 or phrase.upper() in stopwords:
                continue
            if is_common_english_word(phrase, zipf):
                continue
            acr = full_to_acr.get(phrase.lower())
            term = f"{acronym_map[acr]} ({acr})" if acr else phrase
            term_index[term].add(page_num)

    return term_index


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
        help="Book number in course book series (e.g. SEC565.1 is book number 1, and SEC565.4 is book number 4)",
        type=int,
    )
    ap.add_argument(
        "--cache",
        default=DEFAULT_CACHE,
        dest="cache",
        help="Path to cache directory",
    )
    ap.add_argument(
        "--confidence",
        default=DEFAULT_OCR_CONFIDENCE,
        dest="confidence",
        help="Drop OCR words below this confidence 0-100. "
        "Adjust to not detect watermarks or if your output is jumbled.",
        type=float,
    )
    ap.add_argument(
        "--dpi",
        default=DEFAULT_DPI,
        dest="dpi",
        help="Render DPI for OCR",
        type=int,
    )
    ap.add_argument(
        "--exclude-terms",
        dest="exclude",
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
        "--max-words",
        default=DEFAULT_STRIP_MAX_LEN,
        dest="max_words",
        help="Maximum length of a repeating word-sequence to strip as header/footer",
        type=int,
    )
    ap.add_argument(
        "--min-words",
        default=DEFAULT_STRIP_MIN_LEN,
        dest="min_words",
        help="Minimum length of a repeating word-sequence to strip as header/footer",
        type=int,
    )
    ap.add_argument(
        "--occurrences",
        default=DEFAULT_OCCURRENCES,
        dest="occurrences",
        help="Minimum distinct pages on which a term must appear",
        type=int,
    )
    ap.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
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
        "--wordlist",
        default=DEFAULT_WORDLIST,
        dest="wordlist",
        help="Directory containing wordlist text files (one term / line). "
        "connectors.txt / forbidden_starters.txt / stopwords.txt",
    )
    ap.add_argument(
        "--zipf",
        default=DEFAULT_ZIPF,
        dest="zipf",
        help="Drops single common-English words scoring above the value on the Zipf frequency scale. "
        "Requires 'pip install wordfreq'; Does not take effect if not installed. "
        "Lower = stricter (drops more borderline words), higher = more permissive.",
        type=float,
    )

    args = ap.parse_args()

    if args.no_cache:
        args.cache = None

    if not args.password:
        args.password = os.environ.get("PDF_PASSWORD")
        if not args.password:
            log_msg("main", "No password provided")

    if args.exclude:
        exclude_set = load_exclude_set(args.exclude)
    else:
        exclude_set = set()

    wordlists = load_wordlists(args.wordlist)

    try:
        pages = get_pages(
            cache_dir=args.cache,
            confidence=args.confidence,
            dpi=args.dpi,
            password=args.password,
            path=args.path,
        )
    except (OSError, ValueError) as e:
        log_msg("main", f"ERROR: {e}")
        sys.exit(1)

    pages_cleaned = strip_repeating_lines(
        max_len=args.max_words,
        min_len=args.min_words,
        pages=pages,
        threshold=args.threshold,
    )

    terms_index = extract_terms(
        exclude_set=exclude_set,
        pages=pages_cleaned,
        wordlists=wordlists,
        zipf=args.zipf,
    )

    rows = build_index(terms_index, args.book, args.occurrences)

    with open(
        args.output, "w" if args.overwrite else "a", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Term", "Book", "Page"])
        writer.writerows(rows)
    log_msg("main", f"Wrote {len(rows)} lines to '{args.output}'")


if __name__ == "__main__":
    main()
