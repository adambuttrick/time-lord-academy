import os
import argparse
import logging
import joblib
from sklearn_crfsuite import CRF
from utils import create_dictionaries, create_training_data_from_xml, sent2features, sent2labels


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Train CRF model for affiliation parsing.')
    parser.add_argument('-t', '--training_data', type=str,
                        default='data/tagged_affiliations.xml', help='Input XML file containing training data')
    parser.add_argument('-o', '--output', type=str, default='model/affiliation_parser_crf_model.joblib',
                        help='Output file to save the trained model')
    parser.add_argument('-c', '--countries', type=str,
                        default='data/countries.txt', help='File containing list of countries')
    parser.add_argument('-n', '--institutions', type=str,
                        default='data/institution_keywords.txt', help='File containing institution keywords')
    parser.add_argument('-d', '--addresses', type=str,
                        default='data/address_keywords.txt', help='File containing address keywords')
    parser.add_argument('--c1', type=float, default=0.1,
                        help='L1 regularization parameter')
    parser.add_argument('--c2', type=float, default=0.1,
                        help='L2 regularization parameter')
    parser.add_argument('--max_iterations', type=int,
                        default=100, help='Maximum number of iterations')
    return parser.parse_args()


def train_crf_model(X_train, y_train, **kwargs):
    crf = CRF(
        algorithm='lbfgs',
        c1=kwargs.get('c1', 0.1),
        c2=kwargs.get('c2', 0.1),
        max_iterations=kwargs.get('max_iterations', 100),
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf



def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)


def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Creating dictionaries...")
    country_dict, institution_dict, address_dict = create_dictionaries(
        args.countries, args.institutions, args.addresses)
    logging.info("Loading and preprocessing training data...")
    train_sents = create_training_data_from_xml(args.training_data)
    X_train = [sent2features(
        s, country_dict, institution_dict, address_dict) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    logging.info("Training CRF model...")
    crf_model = train_crf_model(
        X_train, y_train, c1=args.c1, c2=args.c2, max_iterations=args.max_iterations)
    logging.info(f"Saving model to {args.output}...")
    save_model(crf_model, args.output)
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
