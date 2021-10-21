import argparse
import re
import glob
import os
import json

import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('-o', '--outfile', default=None, help='Path to outfile (pdf or png)')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    results = []

    for fn in glob.glob(os.path.join(dir_path, 'results', '*.json')):
        match = re.match('result_e([0-9]+)-b([0-9]+).json', os.path.basename(fn))
        epochs, batch_size = match.groups()
        epochs, batch_size = int(epochs), int(batch_size)

        with open(fn, 'r') as stream:
            accuracy = json.load(stream)['accuracy']

        results.append([epochs, batch_size, accuracy])

    all_batch_sizes = sorted(set([batch_size for epochs, batch_size, accuracy in results]))
    all_epochs = sorted(set([epochs for epochs, batch_size, accuracy in results]))

    result_matrix = np.zeros((len(all_batch_sizes), len(all_epochs)))
    for epochs, batch_size, accuracy in results:
        result_matrix[all_batch_sizes.index(batch_size), all_epochs.index(epochs)] = accuracy

    plt.imshow(result_matrix, origin='lower')
    plt.colorbar()
    plt.yticks(np.arange(len(all_batch_sizes)), all_batch_sizes)
    plt.xticks(np.arange(len(all_epochs)), all_epochs)
    plt.ylabel('Batch size')
    plt.xlabel('Epochs')
    plt.clim(vmax=1.0)

    plt.tight_layout()

    if args.outfile:
        plt.savefig(args.outfile)
    else:
        plt.show()
