# CRF Fallback Script for Affiliation Parsing

Uses the trained Conditional Random Field (CRF) model from `train_model.py` to parse affiliations in a fallback matching strategy, using the existing ROR matching.

## Installation

`pip install -r requirements.txt`

## Usage

```
python crf_fallback.py -i <input_file.csv> -o <output_file.csv> -m <crf_model.joblib> -c <countries.txt> -n <institutions.txt> -d <addresses.txt> [-v]
```

## Arguments

- `-i, --input`: Input CSV file containing affiliations (required)
- `-o, --output`: Output CSV file for results (default: 'ror-affiliation_results.csv')
- `-m, --model`: Path to the trained CRF model file (default: 'model/affiliation_parser_crf_model.joblib'. Train first using `train_model.py` and add to the script directory.)
- `-c, --countries`: File containing list of countries (default: 'data/countries.txt')
- `-n, --institutions`: File containing institution keywords (default: 'data/institution_keywords.txt')
- `-d, --addresses`: File containing address keywords (default: 'data/address_keywords.txt')
- `-v, --verbose`: Enable verbose logging

## Input Format

The input CSV file should contain a column named 'affiliation' with the affiliation strings to be parsed and queried.

## Output Format

The script will add the following columns to the input CSV:

- `predicted_ror_id`: The ROR ID(s) of the matched organization(s)
- `prediction_score`: The confidence score(s) of the match(es)
- `match_type`: Indicates whether the match was found by the initial query or fallback query
- `fallback_queries`: The queries used in the fallback process, if applicable

## Logging

Script creates a log file with the naming format `YYYYMMDD_HHMMSS_ror-affiliation.log`.
