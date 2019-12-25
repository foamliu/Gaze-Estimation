import tarfile

from utils import extract


def extract_tar(filename):
    print('Extracting {}...'.format(filename))
    with tarfile.open(filename) as tar:
        tar.extractall('data')


if __name__ == "__main__":
    extract('data/imgs.zip')
    extract_tar('data/MPIIGaze.tar.gz')
