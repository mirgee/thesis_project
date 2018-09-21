import os.path
import os

# PROJ_ROOT = os.path.abspath(os.path.join(os.curdir))
PROJ_ROOT = os.getenv('THESIS_ROOT', os.path.abspath(os.path.join(os.curdir)))

DATA_ROOT = os.path.abspath(os.path.join(PROJ_ROOT, 'data'))
RAW_ROOT = os.path.abspath(os.path.join(DATA_ROOT, 'raw'))
PROCESSED_ROOT = os.path.abspath(os.path.join(DATA_ROOT, 'processed'))
LABELED_ROOT = os.path.abspath(os.path.join(DATA_ROOT, 'labeled'))

CHANNEL_NAMES = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

META_COLUMN_NAMES = ['freq', 'RESP_4W', 'RESP_FIN', 'REMISE_FIN', 'AGE', 'SEX', 'M_1',
       'M_4', 'M_F', 'délka léčby', 'lék 1', 'lék 2', 'lék 3', 'lék 4']
META_FILE_NAME = 'DEP-POOL_Final_144.xlsx'
