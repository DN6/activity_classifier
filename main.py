from numerapi.numerapi import NumerAPI


def get_config(path):
    with open(path) as f:
        return json.loads(f.read())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--op')

    return parser.parse_args()


def download_data():

    # set up paths for download of dataset and upload of predictions
    data_path = "./data/numerai/%s" % (conf.get("name"))
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    # download current dataset
    dl_succeeded = napi.download_current_dataset(dest_path=data_path,
                                                 unzip=True)
    print("Download succeeded: " + str(dl_succeeded))


def main():

    args = get_args()
    conf = get_config(args.config)

    # most API calls do not require logging in
    napi = NumerAPI(verbosity="info")

    # log in
    credentials = napi.login(email=os.environ.get(
        "NMR_EMAIL"), password=os.environ.get("NMR_PASSWORD"))
    print(json.dumps(credentials, indent=2))

    if str(args.op).lower() == "download":
        download_data(conf)

    else:
