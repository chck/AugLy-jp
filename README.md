# AugLy-jp

> Data augmentation for Japanese text, built on [AugLy](https://github.com/facebookresearch/AugLy).

[![PyPI version](https://img.shields.io/pypi/v/augly-jp.svg)](https://pypi.org/project/augly-jp/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31115/)
[![Test](https://github.com/chck/AugLy-jp/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/chck/AugLy-jp/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/chck/AugLy-jp/graph/badge.svg)](https://app.codecov.io/gh/chck/AugLy-jp)
[![Ruff](https://img.shields.io/badge/lint%20%26%20format-Ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![ty](https://img.shields.io/badge/type%20checker-ty-261230.svg)](https://docs.astral.sh/ty/)
[![License](https://img.shields.io/github/license/chck/AugLy-jp.svg)](LICENSE)

## Installation

The current version requires Python 3.11. The latest PyPI release predates the
uv migration, so install the current version from the `main` branch until the
next release is published:

```bash
uv add "augly-jp @ git+https://github.com/chck/AugLy-jp.git@main"
```

## Usage

```python
from augly_jp import text as text_augmentations

text = "あらゆる現実をすべて自分のほうへねじ曲げたのだ"
augmented = text_augmentations.replace_synonym_words(text)
print(augmented)
```

| Function | Example output | Description |
| --- | --- | --- |
| `replace_synonym_words` | あらゆる現実をすべて自身のほうへねじ曲げたのだ | Substitutes words using the [Sudachi synonym dictionary](https://github.com/WorksApplications/SudachiDict/blob/develop/docs/synonyms.md) |
| `replace_wordembs_words` | あらゆる現実をすべて関心のほうへねじ曲げたのだ | Substitutes words using word embeddings |
| `replace_fillmask_words` | つまり現実を、未来な未来まで変えたいんだ | Generates substitutions with a masked language model |
| `replace_backtranslation_sentences` | そして、ほかの人たちをそれぞれの道に安置しておられた | Augments text through back translation |

Development instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).

Licensed under the [MIT License](LICENSE). Runtime models and datasets retain their own license terms.
