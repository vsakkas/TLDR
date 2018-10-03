import argparse
from operator import itemgetter

from sklearn.feature_extraction.text import TfidfVectorizer

import words_with_dot


def load_file(file):
    with open(file, 'r') as f:
        text = f.read()
    text = text.replace('\n', '')
    return text


def tfidf_vectorizer(text):
    count_vect = TfidfVectorizer(stop_words='english', lowercase=True, token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')
    sparse_text = count_vect.fit_transform([text])
    sparse_dict = {name: idf for name, idf in zip(
        count_vect.get_feature_names(), sparse_text.toarray()[0]) if len(name) > 1}
    return sparse_dict


def split_to_sentences(text):
    ends_with_dot = False
    sentences = []
    for i, sentence in enumerate(text.split('.')):
        if len(sentence.strip()) <= 1 and i > 0:
            sentences[-1] += sentence + '.'
            if len(sentence.strip()) == 1:  # == 0 means multiple consecutive dots
                ends_with_dot = True
        elif ends_with_dot is True:
            sentences[-1] += sentence + '.'
            ends_with_dot = False
        else:
            sentences.append(sentence + '.')

    sentences_2 = []
    for i, sentence in enumerate(sentences):
        last_word_index = sentence.rfind(' ')
        last_word = sentence[last_word_index:].strip().lower()
        if last_word in words_with_dot.tokens and i > 0:
            sentences_2[-1] += sentence
            ends_with_dot = True
        elif ends_with_dot is True:
            sentences_2[-1] += sentence
            ends_with_dot = False
        else:
            sentences_2.append(sentence)

    return sentences_2


def evaluate_sentences(sentences, sparse_dict):
    se_value = []
    for i, sentence in enumerate(sentences):
        se_value.append(0)
        for word in sentence.split(' '):
            if word in sparse_dict:
                se_value[-1] += sparse_dict[word]

    return se_value


def generate_summary(sentences, se_value, percentage):
    summary_value = 0
    max_value = sum(se_value)

    evaluated_sentences = sorted(zip(se_value, sentences, range(len(sentences))), reverse=True)

    i = 0
    target_value = (percentage / 100) * max_value
    summary_sentences = []
    # While the next sentence will help us get closer to 'target_value' than the value of all current sentences
    while target_value - summary_value >= (summary_value + evaluated_sentences[i][0]) - target_value:
        summary_sentences.append(evaluated_sentences[i])
        summary_value += evaluated_sentences[i][0]
        i += 1
        if i >= len(evaluated_sentences):
            break

    summary = sorted(summary_sentences, key=itemgetter(2))
    summary_value_percentage = summary_value * 100 / max_value

    return summary, summary_value_percentage


def tldr(file, percentage=30):
    text = load_file(file)

    sparse_dict = tfidf_vectorizer(text)

    sentences = split_to_sentences(text)

    se_value = evaluate_sentences(sentences, sparse_dict)

    summary = generate_summary(sentences, se_value, percentage)


def main():
    parser = argparse.ArgumentParser(description='Text summarizing tool made in Python3.')
    parser.add_argument('file', help='Text file that will be summarized.')
    parser.add_argument('-p', '--percentage', type=int, choices=range(1, 101), metavar='[1-100]', required=False,
                        help='Percentage of summary compared to the size of the original text (Default: 30).')
    args = parser.parse_args()

    if args.percentage is None:
        tldr(file=args.file)
    else:
        tldr(file=args.file, percentage=args.percentage)


if __name__ == '__main__':
    main()
