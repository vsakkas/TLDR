import argparse
from operator import itemgetter

from sklearn.feature_extraction.text import TfidfVectorizer

import words_with_dot


def validate_arguments(percentage, mode):
    if type(percentage) is not int:
        raise TypeError('invalid type for percentage: ' + str(type(percentage)))
    elif percentage < 0 or percentage > 100:
        raise ValueError('invalid value for percentage: ' + str(percentage))
    if type(mode) is not str:
        raise TypeError('invalid type for mode: ' + str(type(mode)))
    elif mode != 'value' and mode != 'length':
        raise ValueError('invalid value for mode: ' + str(mode))


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
        elif ends_with_dot is True and i > 0:
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
        elif ends_with_dot is True and i > 0:
            sentences_2[-1] += sentence
            ends_with_dot = False
        else:
            sentences_2.append(sentence)

    return sentences_2


def evaluate_sentences(sentences, sparse_dict):
    sentence_value = []
    for i, sentence in enumerate(sentences):
        sentence_value.append(0)
        for word in sentence.split(' '):
            if word in sparse_dict:
                sentence_value[-1] += sparse_dict[word]

    return sentence_value


def generate_summary(sentences, sentence_value, percentage, mode):
    evaluated_sentences = sorted(zip(sentence_value, sentences, range(len(sentences))), reverse=True)

    if mode == 'length':
        max_value = len(''.join(sentences))
    else:  # mode == 'value'
        max_value = sum(sentence_value)
    target_value = (percentage / 100) * max_value

    summary_value = 0
    summary_sentences = []
    next_sentence_value = len(evaluated_sentences[0][1]) if mode == 'length' else evaluated_sentences[0][0]

    i = 0
    # While the next sentence will help us get closer to 'target_value' than the value of all current sentences
    while target_value - summary_value >= (summary_value + next_sentence_value) - target_value:
        summary_sentences.append(evaluated_sentences[i])
        if mode == 'length':
            summary_value += len(evaluated_sentences[i][1])
        else:  # mode == 'value'
            summary_value += evaluated_sentences[i][0]
        i += 1
        if i >= len(sentences):
            break
        next_sentence_value = len(evaluated_sentences[i][1]) if mode == 'length' else evaluated_sentences[i][0]

    summary_sentences = sorted(summary_sentences, key=itemgetter(2))
    summary_value_percentage = summary_value * 100 / max_value

    return summary_sentences, summary_value_percentage


def tldr(file, percentage=30, mode='value'):
    validate_arguments(percentage, mode)

    text = load_file(file)

    sparse_dict = tfidf_vectorizer(text)

    sentences = split_to_sentences(text)

    sentence_value = evaluate_sentences(sentences, sparse_dict)

    summary_sentences, summary_value_percentage = generate_summary(sentences, sentence_value, percentage, mode)

    summary = ''.join([sentence[1] for sentence in summary_sentences]).strip()

    return summary, summary_value_percentage


def main():
    parser = argparse.ArgumentParser(description='Text summarizing tool made in Python3.')
    parser.add_argument('file', help='Text file that will be summarized.')
    parser.add_argument('-p', '--percentage', type=int, default=30,
                        choices=range(1, 101), metavar='[1-100]', required=False,
                        help='Percentage of summary compared to the size of the original text. (Default: 30)')
    parser.add_argument('-m', '--mode', type=str, default='value',
                        choices=['value', 'length'], metavar='[value|length]', required=False,
                        help='Criterion to use in order to generate a summary for the provided text. \
                        "value" uses the calculated value of each token with the usage of the TFIDF vectorizer. \
                        "Length" uses the length of the text in characters. \
                        Both "value" and "length" must be as close to "percentage" as possible. (Default: value)')
    args = parser.parse_args()

    tldr(file=args.file,
         percentage=args.percentage,
         mode=args.mode)


if __name__ == '__main__':
    main()
