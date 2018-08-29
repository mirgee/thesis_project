.PHONY: all clean

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# To build, we need just the data (for now)
all: data/raw/iris.csv

clean:
	rm -f data/raw/*.csv

data/raw/iris.csv:
	python src/data/download.py $(DATA_URL) $@

