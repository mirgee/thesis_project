from itertools import product
from os.path import split, splitext, join

import numpy as np
import pandas as pd
import keras as k
from braindecode.datautil.signalproc import bandpass_cnt

import measures.algorithms as algos
from config import LABELED_ROOT
from data.data_files import CHANNEL_NAMES, DataKind, files_builder
from models.shallow import shal_model


class Memoize:

    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, data, start, end):
        if not (start,end) in self.memo:
            self.memo[(start, end)] = self.f(data[start:end])
        return self.memo[(start, end)]


def receptive_field_sizes_and_strides(model):
    sizes = []
    strides = []
    r = 0
    j = 1
    for layer in model.layers:
        conf = layer.get_config()
        k = None
        if 'kernel_size' in conf:
            k = conf['kernel_size'][1]
        elif 'pool_size' in conf:
            k = conf['pool_size'][1]
        else:
            continue
        r += (k-1)*j
        j *= conf['strides'][1]
        sizes.append(r)
        strides.append(j)
    return sizes, strides


def get_rf_start_end(sizes, strides, unit_n, layer_n):
    start = strides[layer_n] * unit_n
    end = strides[layer_n] * unit_n + sizes[layer_n]
    return start, end


def preprocess(data):
    # data *= 1e6
    low_cut_hz = 4
    high_cut_hz = None
    return bandpass_cnt(data, low_cut_hz, high_cut_hz, 250, filt_order=3, axis=0)


def compute_nl_measure(model_path, measure='lyap'):
    idxs = pd.MultiIndex.from_product([list(range(1, 134)), ['a', 'b']],
                                      names=['patient', 'trial'])
    cols = ['channel_n', 'feature', 'value', 'layer', 'unit', 'trial']
    df = pd.DataFrame(index=idxs, columns=cols)
    df = df.astype({
        'channel': str,
        'value': float,
        'layer': int,
        'unit': int,
        'filter': int,
    })

    model = k.load_model(model_path)
    sizes, strides = receptive_field_sizes_and_strides(model)

    # contain model name, feature
    model_name, _ = splitext(split(model_path)[0])
    measures_path = join(CORRS_ROOT, '_'.join((model_name, measure)))

    for alg in algos.registered_algos:
        if alg.algo_name == 'measure':
            mem_alg = Memoize(alg)
            break
    else:
        raise Exception(f'Algorithm {measure} not registered.')

    for file in files_builder(DataKind.PROCESSED):
        idx = file.id
        trial = file.trial
        # Do the same TS processing we did to construct the model input
        logging.info(f'Trial {idx}-{trial}...')
        for channel in CHANNEL_NAMES:
            data = preprocess(file.df[channel])
            # Compute values for each start end layer
            logging.info(f'Channel {channel}...')
            for layer_n, layer in enumerate(model.layers):
                shape = layer.shape
                filters = range(shape[-1])
                units = range(shape[1])
                logging.info(f'Layer {layer_n} with {shape[-1]+1} '
                             'filters, each {shape[1]+1} units...')
                # for filter_n, unit_n in product(filters, units):
                for unit_n in units:
                    start, end = get_rf_start_end(sizes, strides,
                                                  unit_n, layer_n)
                    value = mem_alg(data, start, end)
                    df.loc[(idx, trial), 'channel'] = channel
                    df.loc[(idx, trial), 'value'] = value
                    df.loc[(idx, trial), 'layer'] = layer_n
                    df.loc[(idx, trial), 'unit'] = unit_n
                    # df.loc[(idx, trial), 'filter'] = filter_n

        df.to_pickle(measures_path)


# TODO This may have to be model specific?
def rebuild_model(model_path, channel_n):
    # Rebuild trained model, but change the first layer to accept different
    # input size
    old_model = k.load_model(model_path)
    temp_weights = [layer.get_weights() for layer in old_model.layers]

    # Rebuild the exact same but change input size
    input_time_length = old_model.layers[0].input_shape[1]
    inp = k.layers.Input((1, input_time_length, 1), name='input')
    l = inp
    # We don't care about the softmax
    for layer in old_model.layers[1:-1]:
        conf = layer.get_config()
        # Pick what we need from the original layer, change the dims
        change_weights = False
        if 'kernel_size' in conf:
            conf['kernel_size'][0] = 1
            change_weights = True
        new_layer = k.layers.deserialize({
            'class_name': layer.__class__.__name__,
            'config': conf
        })
        new_weights = layer.get_weights()[channel_n] if change_weights else \
            layer.get_weights()
        new_layer.set_weights(new_weights)
        l = new_layer(l)

    new_model = k.models.Model(inp, l)

    return new_model



def compute_correlations(model_path, measures_path):
    """
    Compute pairs of (measure value, unit activation) for each unit in each
    layer, compute their correlation for each unit in each layer.
    """
    df = df.load_pickle(measures_path)
    # We will will have to use 9*2*num_units... Max num of units is
    # 5000 -> 9*2*4*5000*4 = 1.44 GB :/
    idx = pd.MultiIndex(levels=[[]]*4, labels=[[]]*4,
                        names=['l', 'h', 'w', 'f'])
    cols = ['activation', 'value']
    df = pd.DataFrame(index=idx, columns=cols)

    for file in files_builder(DataKind.DIRECT):
        for channel_n, channel in enumerate(CHANNEL_NAMES):
            # TODO Compare each channel's value with the activation,
            # save the pair
            model = rebuild_model(model_path, channel_n)


if __name__ == '__main__':
    model = shal_model()
    print(receptive_field_sizes(model))

    nls = np.zeros(len(model.layers))

