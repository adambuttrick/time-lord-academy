# CRF Fallback Matching

Uses a Conditional Random Field (CRF) model to parse affiliations as part of a multi-step, fallback matching strategy.

## Matching Strategy

1. Query single search in the [Crossref Marple API](https://gitlab.com/crossref/labs/marple)
2. If no match, use CRF model to parse affiliation and generate new fallback queries
3. Query ROR API/multi-search with these fallback queries

## Installation

```
pip install -r requirements.txt
```

## Usage

```
python single_search_crf_fallback.py -i <input_file.csv> -o <output_file.csv> -m <crf_model.joblib> -c <countries.txt> -n <institutions.txt> -d <addresses.txt> [--use-crossref-marple] [-v]
```

## Arguments

- `-i, --input`: Input CSV file containing affiliations (required)
- `-o, --output`: Output CSV file for results (default: 'ror-affiliation_results.csv')
- `-m, --model`: Path to the trained CRF model file (default: 'model/affiliation_parser_crf_model.joblib')
- `-c, --countries`: File containing list of countries (default: 'data/countries.txt')
- `-n, --institutions`: File containing institution keywords (default: 'data/institution_keywords.txt')
- `-d, --addresses`: File containing address keywords (default: 'data/address_keywords.txt')
- `-v, --verbose`: Enable verbose logging

## Input Format

The input CSV file should contain a column named 'affiliation' with the affiliation strings to be parsed and queried.

## Output Format

The output will be the input CSV with the following columns added:

- `predicted_ror_id`: The ROR ID(s) of the matched organization(s)
- `prediction_score`: The confidence score(s) of the match(es)
- `match_type`: Indicates the method used to find the match ('marple', 'crf_fallback', or 'no_match')
- `fallback_queries`: The queries used in the fallback process, if applicable


## Logging

The script creates a log file with the naming format `YYYYMMDD_HHMMSS_ror-affiliation.log`. Use the `-v` flag for more detailed logging.

## Results

Here is the updated markdown table incorporating the metrics from the attached files:

```
| Dataset                                   |   Precision |   Recall |   F1 Score |   F0.5 Score |   Specificity |
|:------------------------------------------|------------:|---------:|-----------:|-------------:|--------------:|
| affiliations-crossref-2024-02-19          |    0.926132 | 0.921802 |   0.923962 |     0.925262 |      0.732612 |
| affiliations-springer-2023-10-31          |    0.941198 | 0.915931 |   0.928392 |     0.936033 |      0.041420 |
| affiliations-matching-logs-2023-09-03     |    0.838889 | 0.820652 |   0.829670 |     0.835177 |      0.819315 |
```





