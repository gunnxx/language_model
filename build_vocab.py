import os
import argparse
import utils

from torchtext import data
from model.data_handler import DataHandler


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='./data/train.csv', help='Training dataset in csv to build vocabulary upon')
parser.add_argument('--param_dir', default='./experiments/base_model', help='Requiring vocab size for particular experiment')


if __name__ == '__main__':
    args = parser.parse_args()

    # Load parameters from json file (require vocab_size)
    json_path = os.path.join(args.param_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    print("Loading dataset ...")
    data_handler = DataHandler(train_file=args.train_file)
    print("- done.")

    print("Building word vocabulary ...")
    data_handler.build_vocab(field=data.Field(),
                             target_path=os.path.join(args.param_dir, 'TEXT.Field'),
                             max_size=params.vocab_size)
    print("- done.")