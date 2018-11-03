import matplotlib.pyplot as plt
import logging
import os
from data.utils import df_from_tdt, get_trial_index

from config import RAW_ROOT, VISUAL_ROOT
from pyunicorn.timeseries.recurrence_plot import RecurrencePlot


def recurrence_plot(input_path=RAW_ROOT, output_path=VISUAL_ROOT):
    logging.info('Going to draw recurrence plots...')

    for file_name in os.listdir(input_path):
        if not (file_name.endswith('.fif') or file_name.endswith('.tdt')):
            logging.info('Skipping file %s' % file_name)
            continue

        file_path = os.path.join(input_path, file_name)
        _, _, trial = get_trial_index(file_name)
        try:
            df_in = df_from_tdt(file_path, 60)
        except IndexError:
            logging.info('Skipping file with insufficient duration')
            continue

        rp = RecurrencePlot(df_in.values, threshold=0.1)
        matrix = rp.distance_matrix([[30], [30]], "euclidean")
        plt.ylim(0, 256)
        plt.imshow(matrix)
        plt.show()


def main():
    logging.basicConfig(level=logging.DEBUG)
    recurrence_plot()


if __name__ == '__main__':
    main()
