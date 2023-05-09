import sys, logging, argparse, os

from global_ import FORMULAS_LST_PATH, TRAIN_LST_PATH, VOCABULARY_FILE_PATH
"""
    Original code created for 'What You Get Is What You See: A Visual Markup Decompiler'
        https://arxiv.org/pdf/1609.04938v1.pdf
    
    Modified by A. Dumas & T. Nguyen for CSC 561 purposes.
"""

# def process_args(args):
#     parser = argparse.ArgumentParser(description='Generate vocabulary file.')

#     parser.add_argument('--data-path', dest='data_path',
#                         type=str, required=True,
#                         help=('Input file containing <img_name> <line_idx> per line. This should be the file used for training.'
#                         ))
#     parser.add_argument('--label-path', dest='label_path',
#                         type=str, required=True,
#                         help=('Input file containing a tokenized formula per line.'
#                         ))
#     parser.add_argument('--output-file', dest='output_file',
#                         type=str, required=True,
#                         help=('Output file for putting vocabulary.'
#                         ))
#     parser.add_argument('--unk-threshold', dest='unk_threshold',
#                         type=int, default=1,
#                         help=('If the number of occurences of a token is less than (including) the threshold, then it will be excluded from the generated vocabulary.'
#                         ))
#     parser.add_argument('--log-path', dest="log_path",
#                         type=str, default='log.txt',
#                         help=('Log file path, default=log.txt' 
#                         ))
#     parameters = parser.parse_args(args)
#     return parameters

def generate_latex_vocab():
    label_path = FORMULAS_LST_PATH
    data_path = TRAIN_LST_PATH

    formulas = open(label_path, encoding='latin-1').readlines()
    vocab = {}

    with open(data_path) as fin:
        for line in fin:
            line_idx, _, _ = line.strip().split()
            line_strip = formulas[int(line_idx)].strip()
            tokens = line_strip.split()
            tokens_out = []
            for token in tokens:
                tokens_out.append(token)
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    vocab_sort = sorted(list(vocab.keys()))
    vocab_out = []

    num_unknown = 0
    for word in vocab_sort:
        # only add tokens that appear more than once to vocab
        if vocab[word] > 1:
            vocab_out.append(word)
        else:
            num_unknown += 1

    vocab = [word for word in vocab_out]

    with open(VOCABULARY_FILE_PATH, 'w') as fout:
        fout.write('\n'.join(vocab))

