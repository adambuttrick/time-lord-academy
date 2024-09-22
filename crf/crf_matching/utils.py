import re
import xml.etree.ElementTree as ET


def tokenize(s):
    token_pattern = re.compile(r"""
        ( [^\W\d_]+     # letters
        | \d+           # digits
        | [^\w\s]       # single other character (not letter, digit, or whitespace)
        )""", re.UNICODE | re.VERBOSE)
    tokens = token_pattern.findall(s)
    return tokens


def extract_features(tokens, i, country_dict, institution_dict, address_dict):
    token = tokens[i]
    features = {
        'bias': 1.0,
        'word.lower()': token.lower(),
        'word.isdigit()': token.isdigit(),
        'word.isupper()': token.isupper(),
        'word.islower()': token.islower(),
        'word.istitle()': token.istitle(),
        'word.isnumber()': token.isdigit(),
        'word.allupper()': token.isupper(),
        'word.alllower()': token.islower(),
        'word.startupper()': token[0].isupper() and token[1:].islower() if len(token) > 1 else False,
        'word.iscountry()': token.lower() in country_dict,
        'word.isinstitution()': token.lower() in institution_dict,
        'word.isaddress()': token.lower() in address_dict,
    }

    # Features for previous tokens
    if i > 0:
        token_prev = tokens[i - 1]
        features.update({
            '-1:word.lower()': token_prev.lower(),
            '-1:word.isdigit()': token_prev.isdigit(),
            '-1:word.isupper()': token_prev.isupper(),
            '-1:word.islower()': token_prev.islower(),
            '-1:word.istitle()': token_prev.istitle(),
            '-1:word.iscountry()': token_prev.lower() in country_dict,
            '-1:word.isinstitution()': token_prev.lower() in institution_dict,
            '-1:word.isaddress()': token_prev.lower() in address_dict,
        })
    else:
        features['BOS'] = True

    if i > 1:
        token_prev2 = tokens[i - 2]
        features.update({
            '-2:word.lower()': token_prev2.lower(),
            '-2:word.isdigit()': token_prev2.isdigit(),
            '-2:word.isupper()': token_prev2.isupper(),
            '-2:word.islower()': token_prev2.islower(),
            '-2:word.istitle()': token_prev2.istitle(),
            '-2:word.iscountry()': token_prev2.lower() in country_dict,
            '-2:word.isinstitution()': token_prev2.lower() in institution_dict,
            '-2:word.isaddress()': token_prev2.lower() in address_dict,
        })

    # Features for next tokens
    if i < len(tokens) - 1:
        token_next = tokens[i + 1]
        features.update({
            '+1:word.lower()': token_next.lower(),
            '+1:word.isdigit()': token_next.isdigit(),
            '+1:word.isupper()': token_next.isupper(),
            '+1:word.islower()': token_next.islower(),
            '+1:word.istitle()': token_next.istitle(),
            '+1:word.iscountry()': token_next.lower() in country_dict,
            '+1:word.isinstitution()': token_next.lower() in institution_dict,
            '+1:word.isaddress()': token_next.lower() in address_dict,
        })
    else:
        features['EOS'] = True

    if i < len(tokens) - 2:
        token_next2 = tokens[i + 2]
        features.update({
            '+2:word.lower()': token_next2.lower(),
            '+2:word.isdigit()': token_next2.isdigit(),
            '+2:word.isupper()': token_next2.isupper(),
            '+2:word.islower()': token_next2.islower(),
            '+2:word.istitle()': token_next2.istitle(),
            '+2:word.iscountry()': token_next2.lower() in country_dict,
            '+2:word.isinstitution()': token_next2.lower() in institution_dict,
            '+2:word.isaddress()': token_next2.lower() in address_dict,
        })
    return features


def create_dictionaries(country_file, institution_file, address_file):
    def read_file_to_set(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return set(line.strip().lower() for line in file if line.strip())
    country_dict = read_file_to_set(country_file)
    institution_dict = read_file_to_set(institution_file)
    address_dict = read_file_to_set(address_file)
    return country_dict, institution_dict, address_dict


def create_training_data_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    training_data = []
    for aff in root.findall('aff'):
        aff_data = []
        for element in aff:
            tag = element.tag
            text = element.text.strip() if element.text else ""
            if text:
                tokens = tokenize(text)
                if tag == 'institution':
                    aff_data.extend((token, 'INSTITUTION') for token in tokens)
                elif tag == 'addr-line':
                    aff_data.extend((token, 'ADDRESS') for token in tokens)
                elif tag == 'country':
                    aff_data.extend((token, 'COUNTRY') for token in tokens)
                # Add a separator token if it's not the last element and there's text
                if element != aff[-1] and aff_data:
                    aff_data.append((',', 'O'))
        if aff_data:
            training_data.append(aff_data)

    return training_data


def sent2features(sent, country_dict, institution_dict, address_dict):
    tokens = [token for token, label in sent]
    return [extract_features(tokens, i, country_dict, institution_dict, address_dict) for i in range(len(tokens))]


def sent2labels(sent):
    return [label for token, label in sent]