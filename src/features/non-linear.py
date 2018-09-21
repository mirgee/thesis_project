import nolds
from utils import df_from_fif
from config import CHANNEL_NAMES



def compute_lyapunov(data):
    pass


def compute_corr_dim(data):
    pass


def compute_dfa(data):
    pass

def compute_higuchi(data):
    pass


def compute_nl(file_name):
    """Compute vector of non-linear features for trial recorded in file_name"""
    df = df_from_fif(file_name)
    features = []

    for channel in CHANNEL_NAMES:
        lyap = compute_lyapunov(df[channel])
        corr = compute_corr_dim(df[channel])
        dfa = compute_dfa(df[channel])
        higu = compute_higuchi(df[channel])

        features.extend(lyap, corr, dfa, higu)

    return features


@click.command()
def main():
    pass

if __name__ == '__main__':
    main()
