# Thesis project

This is the source code of my diploma thesis project.

## Setup
Set `THESIS_ROOT` environment variable to your absolute path to `src/` folder.

## How to run a specific file
Each file run must have information about the parent module, which means that to run just, e.g. `plot_eeg_from_file.main()`, you have to pass `-m` flag to the interpreter, and give the absolute module name:

```
python -m src.visualization.plot_eeg_from_file raw/1a.tdt
```
