# nlpaug-japanese
Data Augmentation for Japanese Text

## Prerequisites
| Software                   | Install (Mac)              |
|----------------------------|----------------------------|
| [Python 3.8.10][python]    | `pyenv install 3.8.10`     |
| [Poetry 1.1.*][poetry]     | `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \| python`|

[python]: https://www.python.org/downloads/release/python-3810/
[poetry]: https://python-poetry.org/

## Get Started
### Installation
```bash
pip install nlpaug-jp
```

Or clone this repository:
```bash
git clone https://github.com/chck/nlpaug-japanese.git
poetry install
```

### Test (includes lint and format)
```bash
poetry run task test
```

### Lint, Format only
```bash
poetry run task lint
poetry run task fmt
```
