import argparse
from operator import itemgetter

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk.data


def load_file(file):
    with open(file, 'r') as f:
        text = f.read()
    text = text.replace('\n', '')
    return text


def tfidf_vectorizer(text):
    count_vect = TfidfVectorizer(stop_words='english', lowercase=True)
    sparse_text = count_vect.fit_transform([text])
    sparse_dict = {name: idf for name, idf in zip(count_vect.get_feature_names(), sparse_text.toarray()[0])}
    return sparse_dict


def split_to_sentences(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    return sentences


def tldr(file, percentage=30):
    text = load_file(file)

    sparse_dict = tfidf_vectorizer(text)

    sentences = split_to_sentences(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Text file that will be summarized.')
    args = parser.parse_args()

    tldr(file=args.file)


if __name__ == '__main__':
    main()
