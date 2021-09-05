# AugLy-jp
> Data Augmentation for **Japanese Text** on AugLy

[![PyPI Version][pypi-image]][pypi-url]
[![Python Version][python-image]][python-image]
[![Python Style Guide][black-image]][black-url]

## Augmenter
`base_text = "あらゆる現実をすべて自分のほうへねじ曲げたのだ"`

Augmenter | Augmented | Description
:---:|:---:|:---:
SynonymAugmenter|あらゆる現実をすべて自身のほうへねじ曲げたのだ|Substitute similar word according to [Sudachi synonym](https://github.com/WorksApplications/SudachiDict/blob/develop/docs/synonyms.md)
WordEmbsAugmenter|あらゆる現実をすべて関心のほうへねじ曲げたのだ|Leverage word2vec, GloVe or fasttext embeddings to apply augmentation
FillMaskAugmenter|つまり現実を、未来な未来まで変えたいんだ|Using masked language model to generate text

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

## License
This software includes the work that is distributed in the Apache License 2.0 [[1]][apache1-url].

[pypi-image]: https://badge.fury.io/py/augly-jp.svg
[pypi-url]: https://badge.fury.io/py/augly-jp
[python-image]: https://img.shields.io/pypi/pyversions/augly-jp.svg
[black-image]: https://img.shields.io/badge/code%20style-black-black
[black-url]: https://github.com/psf/black
[apache1-url]: https://github.com/cl-tohoku/bert-japanese/blob/v2.0/LICENSE
