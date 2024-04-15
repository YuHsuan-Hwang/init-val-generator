# init-val-generator

Generating initial values for 2D Gaussian fitting.

To automatically format the code:
```
black .
```

To apply type checks:
```
mypy -p init_val_generator --strict
```

To run the unit tests:
```
pytest
```

Development build:
```
pip install -e .
```

Production build:
```
pip install .
```

To build documentation html files:
```
cd docs
make html
```

To store conda env:
```
conda env export --no-builds | grep -v "prefix" > environment.yml
```

To restore conda env:
```
conda env create -f environment.yml
```