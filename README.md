# AugLy-jp
Data Augmentation for Japanese Text on AugLy

## Prerequisites
| Software                   | Install (Mac)              |
|----------------------------|----------------------------|
| [Python 3.8.11][python]    | `pyenv install 3.8.11`     |
| [Poetry 1.1.*][poetry]     | `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \| python`|

[python]: https://www.python.org/downloads/release/python-3811/
[poetry]: https://python-poetry.org/

## Get Started
### Installation
```bash
pip install augly_jp
```

Or clone this repository:
```bash
git clone https://github.com/chck/AugLy-jp.git
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

## Inspired
- https://github.com/facebookresearch/AugLy
- https://github.com/makcedward/nlpaug
- https://github.com/QData/TextAttack
