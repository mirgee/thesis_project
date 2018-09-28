# Thesis project

This is the source code of my diploma thesis project.

## Setup
Add absolute path to `thesis_project/src/` folder to your `PYTHONPATH` env variable.
```
export PYTHONPATH=$PYTHONPATH:{full_path_to_repo}/thesis_project/src/
```

Also, copy the `.tdt` files and excel metadata to the `thesis_project/data/raw`
folder.

Install dependencies into a conda environment:
```
conda env update -f thesis.yml
conda activate thesis
pip3 install -r requirements.txt
```

## How to run a specific file
Each file run must have information about the parent module, which means that to run just, e.g. `plot_eeg_from_file.main()`, you have to pass `-m` flag to the interpreter, and give the absolute module name:

```
python -m src.visualization.interactive_plot_eeg_from_file raw/1a.tdt
```

Alternatively, after adding the source code root to `PYTHONPATH` and activating
conda environment with all the dependencies installed, you can just run:

```
python src/visualization/interactive_plot_eeg_from_file raw/1a.tdt
```
