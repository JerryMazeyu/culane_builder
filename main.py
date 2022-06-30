from dataset_builder import CULaneTrainBuilder, CULaneTestBuilder
import argparse
import pprint
import pickle as pkl


def main():
    args = parse_args()
    if args.train:
        Builder = CULaneTrainBuilder if args.train else CULaneTestBuilder
    builder = Builder(args.path, args.type, args.cache)
    builder.main()
    if args.check:
        with open(args.cache, 'rb') as cache_file:
            data_infos = pkl.load(cache_file)
            for x in data_infos:
                x['lanes'] = 'xxx, ' * len(x['lanes'])
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(data_infos)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--train', default=True, help='Train(True) or Test(False)')
    parser.add_argument('--path', help='If train, path is json file path or json path, otherwise is image path')
    parser.add_argument('--type', default='a', choices=['a', 'w'], help='a is append and w is write, write will cover original cache.')
    parser.add_argument('--cache', help='Output pickle file path, if append, it should be original pickle file.')
    parser.add_argument('--check', default=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()