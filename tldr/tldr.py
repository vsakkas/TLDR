import argparse
from operator import itemgetter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import words_with_dot


def _validate_arguments(percentage, mode):
    """
    Validates the type and value of the provided parameters.
    Parameters
    ----------
    percentage : int
        Percentage parameter of the `tldr` function. Must be an int between 0 and 100.
    mode : str
        Mode paramater of the `tldr` function. Must have on the following values: "value", "length" and "best".
    Raises
    -------
    TypeError
        If the type of the parameters is incorrect.
    ValueError
        If the value of the parameters is incorrect.
    """
    if type(percentage) is not int:
        raise TypeError('invalid type for percentage: ' + str(type(percentage)))
    elif percentage < 0 or percentage > 100:
        raise ValueError('invalid value for percentage: ' + str(percentage))
    if type(mode) is not str:
        raise TypeError('invalid type for mode: ' + str(type(mode)))
    elif mode != 'value' and mode != 'length' and mode != 'best':
        raise ValueError('invalid value for mode: ' + str(mode))


def _load_file(file):
    """
    Loads and returns file from given path as a single string.
    Newlines are removed from the string.
    Parameters
    ----------
    file : str
        Path of the file to load.
    Returns
    -------
    str
    """
    with open(file, 'r') as f:
        text = f.read()
    text = text.replace('\n', '')
    return text


def _tfidf_vectorizer(text, vocabulary):
    """
    Uses TFIDF vectorizer to create a dictionary of words and their value.
    If `vocabulary` is not None, then it is used to generate a vocabulary.
    If no file is provided, the file to be summarized will be used instead
    in order to extract a vocabulary, although this is not suggested.
    All numbers and English stop words are removed from the text and all
    words are converted to lowercase.
    Parameters
    ----------
    text : str
        Text that will be used to generate a dictionary of words and their value.
    vocabulary : str
        Path of CSV file that will be used to extract the vocabulary.
        Must contain one column named "content".
    Returns
    -------
    dict
        Sparse dictionary that contains words from the provided text along with their value.
    """
    count_vect = TfidfVectorizer(stop_words='english', lowercase=True, token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')
    if vocabulary is None:
        sparse_text = count_vect.fit_transform([text])
    else:
        df = pd.read_csv(vocabulary)
        count_vect.fit(df['content'])
        sparse_text = count_vect.transform([text])
    sparse_dict = {name: idf for name, idf in zip(
        count_vect.get_feature_names(), sparse_text.toarray()[0]) if len(name) > 1}
    return sparse_dict


def _split_to_sentences(text):
    """
    Receives a string and generates a list of sentences.
    Works in two phases in order to avoid certain ege cases where
    splitting based on the dot character is not enough. First, it
    separates the string based on the dor character while handling
    multiple consecutive dot characters and adding them to the
    previous sentence. Secondly, it uses the `words_with_dot.py` file
    in order to find words that end with a dot the seperated strings
    into correct sentences.
    Parameters
    ----------
    text : str
        Text that will be separated into sentences.
    Returns
    -------
    list
        List of strings. Each string is a sentence.
    """
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

    # combine incorrectly split sentences by using a dictionary
    # of words that end with a dot
    split_sentences = []
    for i, sentence in enumerate(sentences):
        last_word_index = sentence.rfind(' ')
        last_word = sentence[last_word_index:].strip().lower()
        if last_word in words_with_dot.tokens and i > 0:
            split_sentences[-1] += sentence
            ends_with_dot = True
        elif ends_with_dot is True and i > 0:
            split_sentences[-1] += sentence
            ends_with_dot = False
        else:
            split_sentences.append(sentence)

    return split_sentences


def _evaluate_sentences(sentences, sparse_dict):
    """
    Uses a list of sentences and a dictionary with words and
    their value to find the total value per sentence.
    Parameters
    ----------
    sentences : list
        List of sentences that will be used evaluated.
    sparse_dict: dict
        Dictionary of words and their value that will be used to evaluate the sentence.
    Returns
    -------
    list
        List of floating point numbers. Contains the value of each sentence.
    """
    sentence_value = []
    for i, sentence in enumerate(sentences):
        sentence_value.append(0)
        for word in sentence.split(' '):
            if word in sparse_dict:
                sentence_value[-1] += sparse_dict[word]

    return sentence_value


def _generate_summary(sentences, sentence_value, percentage, mode):
    """
    Generates a summary from provided text by using the list of `sentences`,
    the `sentence_value` and the appropriate `mode`.
    Parameters
    ----------
    sentences : list
        List of sentences that will be used to generate the summary.
    sentence_value : list
        List of the value of each sentence.
    percentage: int
        Percentage of the desired value or length of the summary. Ignored when mode is 'best'.
    mode : str
        Mode that determines how the summary will be generated. (described at the `tldr` function)
    Returns
    -------
    list
        List of strings. It's the generated summary separated into sentences.
    double
        The percentage of the value of the summary. If mode is 'best' or 'value' then
        it refers to the value of the summary in comparison to the total value of the
        original text. If mode is 'length' then it refers to the length of the summary
        in comparison to the length of the original text.
    """
    # Combine the sentences with their value and their position on the original text
    evaluated_sentences = sorted(zip(sentence_value, sentences, range(len(sentences))), reverse=True)

    if mode == 'length':
        max_value = len(''.join(sentences))
    else:  # mode == 'value' or mode == 'best'
        max_value = sum(sentence_value)

    if mode == 'best':
        # set a high enough value so that the while statement will not stop when checking the target_value
        target_value = max_value + 1
    else:  # mode == 'value' or mode == 'length'
        target_value = (percentage / 100) * max_value

    average_value = sum(sentence[0] for sentence in evaluated_sentences) / len(evaluated_sentences)

    summary_value = 0
    summary_sentences = []
    next_sentence_value = len(evaluated_sentences[0][1]) if mode == 'length' else evaluated_sentences[0][0]

    i = 0
    # While the next sentence will help us get closer to 'target_value' than the value of all current sentences
    while target_value - summary_value >= (summary_value + next_sentence_value) - target_value:
        if mode == 'best' and evaluated_sentences[i][0] < average_value:
            break
        summary_sentences.append(evaluated_sentences[i])
        if mode == 'length':
            summary_value += len(evaluated_sentences[i][1])
        else:  # mode == 'value' or mode == 'best'
            summary_value += evaluated_sentences[i][0]
        i += 1
        if i >= len(sentences):
            break
        next_sentence_value = len(evaluated_sentences[i][1]) if mode == 'length' else evaluated_sentences[i][0]

    # Sort sentences based on their original order
    summary_sentences = sorted(summary_sentences, key=itemgetter(2))
    summary_value_percentage = summary_value * 100 / max_value

    return summary_sentences, summary_value_percentage


def tldr(file, percentage=30, mode='best', vocabulary=None):
    """
    Opens provided file and generates a summary.
    Parameters
    ----------
    file : string
        File that contains the text to be summarized.
    percentage : int
        Percentage of length or value summary compared to the length or value of the original text. (default: 30)
    mode : str
        Criterion to use in order to generate a summary for the provided text. (default: 'best')
        * "value" uses the calculated value of each token with the usage of the TFIDF vectorizer.
        * "length" uses the length of the text in characters.
        * "best" uses the calculated value of each token and only selects the sentences witha value that is higher
        than the average calculated value per sentence.
        When "value" or "length" is selected, the value/length of the summary must be as close to the provided "percentage" parameter as possible.
        When "best" mode is selected, the value of `percentage` is ignored.
    Returns
    -------
    str
        The generated summary.
    double
        The percentage of the value of the summary. If mode is 'best' or 'value' then
        it refers to the value of the summary in comparison to the total value of the
        original text. If mode is 'length' then it refers to the length of the summary
        in comparison to the length of the original text.
    """
    _validate_arguments(percentage, mode)

    text = _load_file(file)

    sparse_dict = _tfidf_vectorizer(text, vocabulary)

    sentences = _split_to_sentences(text)

    sentence_value = _evaluate_sentences(sentences, sparse_dict)

    summary_sentences, summary_value_percentage = _generate_summary(sentences, sentence_value, percentage, mode)

    summary = ''.join([sentence[1] for sentence in summary_sentences]).strip()

    return summary, summary_value_percentage


def main():
    parser = argparse.ArgumentParser(description='Text summarizing tool made in Python3.')
    parser.add_argument('file', help='Text file that will be summarized.')
    parser.add_argument('-p', '--percentage', type=int, default=30,
                        choices=range(1, 101), metavar='[1-100]', required=False,
                        help='Percentage of summary compared to the size of the original text. (default: 30)')
    parser.add_argument('-m', '--mode', type=str, default='best',
                        choices=['value', 'length', 'best'], metavar='[value|length|best]', required=False,
                        help='Criterion to use in order to generate a summary for the provided text. \
                        "value" uses the calculated value of each token with the usage of the TFIDF vectorizer. \
                        "length" uses the length of the text in characters. \
                        "best" uses the calculated value of each token and only selects the sentences \
                        with a value that is higher than the average calculated value per sentence. \
                        When "value" or "length" is selected, the value/length of the summary must \
                        be as close to the provided "percentage" parameter as possible. (default: best)')
    parser.add_argument('-v', '--vocabulary', type=str, required=False,
                        help='CSV file that contains columns of text that will be used to extract a vocabulary \
                        using the TFIDF vectorizer. If no file is provided, the file to be summarized \
                        will be used instead in order to extract a vocabulary, although this is not suggested.')
    args = parser.parse_args()

    summary = tldr(file=args.file,
                   percentage=args.percentage,
                   mode=args.mode,
                   vocabulary=args.vocabulary)
    print(summary[0])


if __name__ == '__main__':
    main()
