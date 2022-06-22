import argparse

parser = argparse.ArgumentParser(description = 'TransferLearning')


parser.add_argument(
    '--train_batch',
    default = '10',
    type = int,
    help = 'Training batch size'
)
parser.add_argument(
    '--val_batch',
    default = '10',
    type = int,
    help = 'Validation batch size'
)
parser.add_argument(
    '--output_classes',
    default = '2',
    type = int,
    help = 'Number of output classes'
)
parser.add_argument(
    '--epochs',
    default = '6',
    type = int,
    help = 'Number of epochs for training'
)
args = parser.parse_args()