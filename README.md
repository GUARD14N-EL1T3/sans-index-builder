# sans-index-builder

## Requirements

- Tesseract OCR binary (easy on linux/WSL, more setup on native Windows)
    - [Tesseract Installation Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- Python 3 (tested on v3.14.7)
- Pip
    - `python -m ensurepip --upgrade`
- Packages found in `requirements.txt`
    - `pip install -r requirements.txt`

## Usage

Basic usage: `python index_builder.py for565-book-1.pdf 1`

Will create `index.csv` and __append__ to the file, uses all the defaults for rasterization and OCR text extraction, will attempt to strip the header/footer information by detecting the phrases across the pages. Eventually, `index.csv` will have the data from books 1-5 appended in the file, and you can utilize the `sans-index-formatter` to produce a PDF with a glossary index.

```
usage: index_builder.py [-h] [--cache CACHE] [--confidence CONFIDENCE] [--dpi DPI] [--exclude-terms EXCLUDE] [--no-cache] [--max-words MAX_WORDS] [--min-words MIN_WORDS] [--occurrences OCCURRENCES] [--output OUTPUT] [--overwrite] [--password PASSWORD] [--threshold THRESHOLD] [--wordlist WORDLIST] [--zipf ZIPF] path book

positional arguments:
  path                  Path to target PDF
  book                  Book number in course book series (e.g. SEC565.1 is book number 1, and SEC565.4 is book number 4)

options:
  -h, --help            show this help message and exit
  --cache CACHE         Path to cache directory (default: cache)
  --confidence CONFIDENCE
                        Drop OCR words below this confidence 0-100. Adjust to not detect watermarks or if your output is jumbled. (default: 40)
  --dpi DPI             Render DPI for OCR (default: 300)
  --exclude-terms EXCLUDE
                        Path to a file of terms/phrases to always exclude (one term / line), include generic boilerplate or other garbage not needed ever. Applied in both 'build' and 'list' modes. (default: None)
  --no-cache            Ignore cache option flags and force OCR text extraction (default: False)
  --max-words MAX_WORDS
                        Maximum length of a repeating word-sequence to strip as header/footer (default: 10)
  --min-words MIN_WORDS
                        Minimum length of a repeating word-sequence to strip as header/footer (default: 3)
  --occurrences OCCURRENCES
                        Minimum distinct pages on which a term must appear (default: 2)
  --output OUTPUT       Output path for CSV file containing index (default: index.csv)
  --overwrite           Overwrite a potentially existing output file path (default: False)
  --password PASSWORD   Encrypted PDF password and can be set using the environment variable 'PDF_PASSWORD' (default: None)
  --threshold THRESHOLD
                        Ratio of pages that will trigger stripping a word-sequence (default: 0.5)
  --wordlist WORDLIST   Directory containing wordlist text files (one term / line). connectors.txt / forbidden_starters.txt / stopwords.txt (default: wordlists)
  --zipf ZIPF           Drops single common-English words scoring above the value on the Zipf frequency scale. Requires 'pip install wordfreq'; Does not take effect if not installed. Lower = stricter (drops more borderline words), higher = more permissive. (default: 3.6)
```

## Disclaimer

Claude was used to make the initial draft of the script. Script was subsequently reviewed line-by-line and adjusted to make it more human-readable and fit my coding style.
