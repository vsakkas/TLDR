import argparse


def tldr(file):
    with open(file, 'r') as f:
        text = f.read()
    text = text.replace('\n', '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Text file that will be summarized.')
    args = parser.parse_args()

    tldr(file=args.file)


if __name__ == '__main__':
    main()
