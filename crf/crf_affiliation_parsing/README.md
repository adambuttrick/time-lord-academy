# Train Affiliation Parser Model

Scripts for training a Conditional Random Field (CRF) model to parse  affiliations.

## Requirements

`pip install -r requirements.txt`

## Usage

```
python train_model.py -t <training_data.xml> -o <output_model.joblib> -c <countries.txt> -n <institutions.txt> -d <addresses.txt>
```

## Arguments

- `-t, --training_data`: Input XML file containing training data (default: 'data/tagged_affiliations.xml')
- `-o, --output`: Output file to save the trained model (default: 'model/affiliation_parser_crf_model.joblib')
- `-c, --countries`: File containing list of countries (default: 'data/countries.txt')
- `-n, --institutions`: File containing institution keywords (default: 'data/institution_keywords.txt')
- `-d, --addresses`: File containing address keywords (default: 'data/address_keywords.txt')
- `--c1`: L1 regularization parameter (default: 0.1)
- `--c2`: L2 regularization parameter (default: 0.1)
- `--max_iterations`: Maximum number of iterations (default: 100)


## Output
The trained model is saved as a joblib file, which can be loaded to parse affiliations.