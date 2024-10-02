import re
import csv
import logging
import argparse
import joblib
import requests
import time
from datetime import datetime
from urllib.parse import quote
from utils import tokenize, create_dictionaries, extract_features


def setup_logging(verbose):
    now = datetime.now()
    script_start = now.strftime("%Y%m%d_%H%M%S")
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(filename=f'{script_start}_ror-affiliation.log', level=log_level,
                        format='%(asctime)s %(levelname)s %(message)s')


def query_marple(affiliation, verbose, strategy="affiliation-single-search"):
    results = []
    start_time = time.time()
    try:
        base_url = "https://marple.research.crossref.org/match"
        params = {
            "task": "affiliation-matching",
            "input": quote(affiliation),
            "strategy": strategy
        }
        url = f"{base_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        r = requests.get(url)
        api_response = r.json()

        if api_response["status"] == "ok":
            items = api_response["message"]["items"]
            for item in items:
                ror_id = item["id"]
                confidence = item["confidence"]
                results.append((ror_id, confidence))
                if verbose:
                    logging.debug(f"Crossref Marple match found for '{affiliation}': {ror_id} (confidence: {confidence})")
                break
        if not results and verbose:
            logging.debug(f"No Crossref Marple match found for '{affiliation}'")
    except Exception as e:
        logging.error(f'Error in Crossref Marple query: {affiliation} - {e}')
    elapsed_time = time.time() - start_time
    return results, elapsed_time


def parse_affiliation(s, crf_model, country_dict, institution_dict, address_dict, verbose):
    tokens = tokenize(s)
    features = [extract_features(
        tokens, i, country_dict, institution_dict, address_dict) for i in range(len(tokens))]
    labels = crf_model.predict_single(features)
    if verbose:
        logging.debug(f"Parsed affiliation: {list(zip(tokens, labels))}")
    result = []
    current_label = labels[0]
    current_tokens = [tokens[0]]
    for token, label in zip(tokens[1:], labels[1:]):
        if label == current_label:
            current_tokens.append(token)
        else:
            result.append((current_label, ' '.join(current_tokens)))
            current_label = label
            current_tokens = [token]
    result.append((current_label, ' '.join(current_tokens)))
    institutions = [text for label, text in result if label == 'INSTITUTION']
    addresses = [text for label, text in result if label == 'ADDRESS']
    countries = [text for label, text in result if label == 'COUNTRY']
    return institutions, addresses, countries


def normalize_punctuation(text):
    return re.sub(
        r'\s*([.,!?])\s*|\s+|(\w)\s*\.\s*(\w)\s*\.\s*|\s\'(?=\s)',
        lambda m: (m.group(1) + ' ' if m.group(1) else
                   '.' if m.group(2) else
                   "'" if m.group(0).strip() == "'" else
                   ' '),
        text
    ).strip()


def execute_fallback_query(affiliation, crf_model, country_dict, institution_dict, address_dict, verbose):
    results = []
    fallback_queries = []
    total_time = 0
    try:
        institutions, _, countries = parse_affiliation(
            affiliation, crf_model, country_dict, institution_dict, address_dict, verbose)
        country = countries[0] if countries else ""
        for institution in institutions:
            normalized_institution = normalize_punctuation(institution)
            normalized_country = normalize_punctuation(country)
            query = f"{normalized_institution}, {normalized_country}".strip(", ")
            fallback_queries.append(query)
            query_results, query_time = query_marple(
                query, verbose, strategy="affiliation-multi-search")
            total_time += query_time
            results.extend(query_results)
            if results:
                break
        if verbose:
            logging.debug(f"Fallback queries executed for: {affiliation}")
            logging.debug(f"Fallback query texts: {'; '.join(fallback_queries)}")
            if results:
                logging.debug(f"Fallback query successful: {results}")
            else:
                logging.debug("Fallback queries unsuccessful")
    except Exception as e:
        logging.error(f'Error in fallback query: {affiliation} - {e}')
    return results, '; '.join(fallback_queries), total_time


def parse_and_query(input_file, output_file, crf_model, country_dict, institution_dict, address_dict, verbose):
    total_time = 0
    request_count = 0
    try:
        with open(input_file, 'r+', encoding='utf-8-sig') as f_in, open(output_file, 'w') as f_out:
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames + [
                "predicted_ror_id", "prediction_score",
                "match_type", "fallback_queries"
            ]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                affiliation = row['affiliation']
                results = []
                match_type = "no_match"
                fallback_queries = ""
                results, query_time = query_marple(
                    affiliation, verbose, strategy="affiliation-single-search")
                total_time += query_time
                request_count += 1
                if results:
                    match_type = "marple"
                if not results:
                    results, fallback_queries, fallback_time = execute_fallback_query(
                        affiliation, crf_model, country_dict, institution_dict, address_dict, verbose)
                    total_time += fallback_time
                    if results:
                        match_type = "crf_fallback"
                if results:
                    predicted_ids = ";".join([r[0] for r in results])
                    prediction_scores = ";".join([str(r[1]) for r in results])
                else:
                    predicted_ids = None
                    prediction_scores = None

                row.update({
                    "predicted_ror_id": predicted_ids,
                    "prediction_score": prediction_scores,
                    "match_type": match_type,
                    "fallback_queries": fallback_queries
                })
                writer.writerow(row)
                if verbose:
                    logging.debug(f"Processed affiliation: {affiliation}, Match type: {match_type}, Fallback queries: {fallback_queries}")

        average_time = total_time / request_count if request_count > 0 else 0
        logging.info(f"Average time per request: {average_time:.2f} seconds")
    except Exception as e:
        logging.error(f'Error in parse_and_query: {e}')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Return ROR affiliation matches for a given CSV file.')
    parser.add_argument('-i', '--input', help='Input CSV file', required=True)
    parser.add_argument('-o', '--output', help='Output CSV file',
                        default='ror-affiliation_results.csv')
    parser.add_argument('-m', '--model', help='Path to the trained CRF model file',
                        default='model/affiliation_parser_crf_model.joblib')
    parser.add_argument(
        '-c', '--countries', help='File containing list of countries', default='data/countries.txt')
    parser.add_argument('-n', '--institutions', help='File containing institution keywords',
                        default='data/institution_keywords.txt')
    parser.add_argument('-d', '--addresses', help='File containing address keywords',
                        default='data/address_keywords.txt')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    logging.info("Loading CRF model and dictionaries...")
    crf_model = joblib.load(args.model)
    country_dict, institution_dict, address_dict = create_dictionaries(
        args.countries, args.institutions, args.addresses)
    logging.info("Starting affiliation parsing and querying...")
    parse_and_query(args.input, args.output, crf_model,
                    country_dict, institution_dict, address_dict, args.verbose)
    logging.info("Affiliation parsing and querying completed.")


if __name__ == '__main__':
    main()
