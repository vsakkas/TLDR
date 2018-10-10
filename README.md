# TL;DR
Text summarizing tool made in Python3. Uses a TF-IDF vectorizer to evaluate the "value" of each sentence in a provided file and constructs a summary using the most "important" sentences based on 3 different criteria which are described below.

## Getting Started

To get this project running on your local machine, follow the following instructions.

### Prerequisites

First, you need to download `scikit-learn` (version 0.20.0 and newer) and `pandas` (version 0.23 and newer) using the following command:

```pip3 install -U scikit-learn pandas```

*Note: make sure python3 (version 3.5 and newer) is installed.*

### Downloading

To download the source code of this project use the following command:

```git clone https://github.com/vsakkas/TLDR.git```

And to enter the directory of the downloaded project, simply type:

```cd TLDR```

### Running

To generate a summary for a file, simply run the following command:

```python3 tldr/tldr.py <file>```

This will read `<file>` and print its summary.

TL;DR supports additional, optional commands. You can read more about by typing:

```python3 tldr/tldr.py -h``` (or `--help`)

Which shows information about the following options:

```
-p [1-100], --percentage [1-100]
                        Percentage of summary compared to the size of the
                        original text. (default: 30)

-m [value|length|best], --mode [value|length|best]
                        Criterion to use in order to generate a summary for
                        the provided text. "value" uses the calculated value
                        of each token with the usage of the TFIDF vectorizer.
                        "length" uses the length of the text in characters.
                        "best" uses the calculated value of each token and
                        only selects the sentences with a value that is higher
                        than the average calculated value per sentence. When
                        "value" or "length" is selected, the value/length of
                        the summary must be as close to the provided
                        "percentage" parameter as possible. (default: best)

-v VOCABULARY, --vocabulary VOCABULARY
                        CSV file that contains columns of text that will be
                        used to extract a vocabulary using the TFIDF
                        vectorizer. If no file is provided, the file to be
                        summarized will be used instead in order to extract a
                        vocabulary, although this is not suggested.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
