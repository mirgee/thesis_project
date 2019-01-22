import os.path
import os

# PROJ_ROOT = os.path.abspath(os.path.join(os.curdir))
PROJ_ROOT = os.getenv('THESIS_ROOT')
DATA_ROOT = os.path.abspath(os.path.join(PROJ_ROOT, 'data'))
RAW_ROOT = os.path.abspath(os.path.join(DATA_ROOT, 'raw'))
PROCESSED_ROOT = os.path.abspath(os.path.join(DATA_ROOT, 'processed'))
LABELED_ROOT = os.path.abspath(os.path.join(DATA_ROOT, 'labeled'))
VISUAL_ROOT = os.path.abspath(os.path.join(DATA_ROOT, 'visual'))
DURATIONS_ROOT = os.path.abspath(os.path.join(DATA_ROOT, 'durations'))
SURROGATE_ROOT = os.path.abspath(os.path.join(DATA_ROOT, 'surrogates'))


CHANNEL_NAMES = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
