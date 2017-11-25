import utils.utils

def get_config(path):
    with open(path) as f:
        return json.loads(f.read())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')

    return parser.parse_args()

def preprocess(conf):
    data = utils.data()




def main():
    args = get_args()
    conf = get_config(args.config)

    preprocess(conf)

if __name__ == '__main__':
    main()
