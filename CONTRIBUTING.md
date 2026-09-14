# Contributing

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.13 or later

## Development setup

```bash
git clone https://github.com/chck/AugLy-jp.git
cd AugLy-jp
uv sync --locked --dev
uv run pre-commit install
```

## Checks

Run the same lint and type checks used by CI:

```bash
uv run task lint
```

Run the test suite:

```bash
uv run pytest tests
```

Format the Python package before submitting a change:

```bash
uv run task fmt
```

Verify packaging changes with:

```bash
uv lock --check
uv build --no-sources
```

## Project context

AugLy-jp builds on ideas and APIs from these projects:

- [AugLy](https://github.com/facebookresearch/AugLy)
- [nlpaug](https://github.com/makcedward/nlpaug)
- [TextAttack](https://github.com/QData/TextAttack)

## Third-party models and data

The project does not relicense models or datasets fetched at runtime. Check the relevant model card or dataset documentation before redistributing those artifacts. In particular, the default fill-mask model is documented by [Tohoku NLP](https://huggingface.co/tohoku-nlp/bert-base-japanese-v2).
