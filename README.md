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

### Test with reformat
```bash
poetry run task test
```

### Reformat
```bash
poetry run task fmt
```

### Lint
```bash
poetry run task lint
```

## Inspired
- https://github.com/facebookresearch/AugLy
- https://github.com/makcedward/nlpaug
- https://github.com/QData/TextAttack

## References
[1] Masatoshi Suzuki, Koji Matsuda, Satoshi Sekine, Naoaki Okazaki and Kentaro Inui. A Joint Neural Model for Fine-Grained Named Entity Classification of Wikipedia Articles. IEICE Transactions on Information and Systems, Special Section on Semantic Web and Linked Data, Vol. E101-D, No.1, pp.73-81, 2018.
