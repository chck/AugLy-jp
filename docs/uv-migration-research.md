# Poetry から uv への移行調査

調査日: 2026-09-14

## 結論

- Astral は現時点で Poetry からの専用移行ガイドを提供しておらず、他ツールから uv への移行ガイドは未提供としている。そのため、`[tool.poetry]` を標準の `[project]`（PEP 621）へ手動変換するのが公式仕様に沿った経路になる。`dependencies` は PEP 508 文字列、開発依存は標準の `[dependency-groups]` に置く。[uv migration guides](https://docs.astral.sh/uv/guides/migration/) / [pyproject.toml specification](https://packaging.python.org/specifications/declaring-project-metadata/) / [uv dependency fields](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- 直接依存をすべて最新安定版に上げる場合、NumPy 2.5.3 の要件により Python は最低 3.12 が必要。ただし後述の GiNZA ELECTRA 制約で選ばれる `tokenizers 0.13.3` は Python 3.11 までの wheel しか提供しないため、今回の検証済み互換 window は `requires-python = ">=3.11,<3.12"` とし、NumPy は Python 3.11 互換の 2.4.6 を採用した。[tokenizers 0.13.3 files](https://pypi.org/pypi/tokenizers/0.13.3/json)
- ただし、`transformers` の最新 5.17.0 は同時採用できない。`ja-ginza-electra 5.2.0` が `spacy-transformers>=1.1.9,<1.2.0` を要求し、許容される `spacy-transformers 1.1.9` が `transformers>=3.4.0,<4.26.0` を要求する。したがって最高候補は `transformers==4.25.1`。[ja-ginza-electra metadata](https://pypi.org/pypi/ja-ginza-electra/5.2.0/json) / [spacy-transformers metadata](https://pypi.org/pypi/spacy-transformers/1.1.9/json) / [transformers 4.25.1 metadata](https://pypi.org/pypi/transformers/4.25.1/json)
- `augly 1.0.0` では `nlpaug==1.1.3` が通常依存から `text` extra へ移った。このコードは `nlpaug` を直接 import するため、`augly[text]>=1.0.0` または `nlpaug==1.1.3` の直接宣言が必要。[augly 0.1.7 metadata](https://pypi.org/pypi/augly/0.1.7/json) / [augly 1.0.0 metadata](https://pypi.org/pypi/augly/1.0.0/json)
- 最新化後も古いパッケージがボトルネックになる。`pysen[lint] 0.12.1` は Black 22.10 以下、Flake8 5 未満、isort 5.2 未満、mypy 0.800 未満を要求する。`dartsclone 0.10.2` は Python 3.9 までの wheel しかなく、Python 3.12 では sdist ビルドになる。`python-magic-bin 0.4.14` は Windows と Intel macOS wheel だけで、macOS arm64 では解決・実行検証が必要。[pysen metadata](https://pypi.org/pypi/pysen/0.12.1/json) / [dartsclone metadata](https://pypi.org/pypi/dartsclone/0.10.2/json) / [python-magic-bin metadata](https://pypi.org/pypi/python-magic-bin/0.4.14/json)
- `chikkarpy 0.1.1` は `dartsclone==0.9.0` を要求するため、直接依存の最新版 0.10.2 よりこの互換 pin が優先される。また build/runtime の両方で未宣言の `pkg_resources` を import するため、setuptools は 80 系に上限を付ける必要がある。

## 推奨する pyproject 構造

Poetry 固有の caret 制約はそのまま PEP 508 では使えない。公開パッケージの依存範囲は `[project].dependencies` に PEP 508 形式で記述し、厳密な解決結果は `uv.lock` に任せる。

```toml
[project]
name = "augly_jp"
version = "2021.9.30"
description = "Data Augmentation for Japanese Text"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
authors = [{ name = "chck", email = "shimekiri.today@gmail.com" }]
dependencies = [
  # PEP 508 requirements
]

[project.urls]
Repository = "https://github.com/chck/AugLy-jp"

[dependency-groups]
dev = [
  # PEP 508 requirements
]
```

`[dependency-groups].dev` は `uv sync` と `uv run` で既定対象になる。uv は全 dependency group をまとめて解決するため、group 間の非互換も lock 時に検出される。古い `[tool.uv].dev-dependencies` は非推奨。[Managing dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)

## lock / sync / run

- `uv lock` は `pyproject.toml` を解決してクロスプラットフォームの `uv.lock` を作る。`uv.lock` は手編集せず、バージョン管理へ含める。既存 lock は新リリースだけでは自動更新されないため、全更新は `uv lock --upgrade`、整合性確認は `uv lock --check` を使う。
- `uv sync` は `.venv` を作成して lock と既定 group を同期し、既定では lock にないパッケージを削除する exact sync。エディタ用環境の明示的構築にも使う。
- `uv run <command>` は実行前に lock と環境を自動更新し、プロジェクト環境でコマンドを実行する。CI で lock の変更を禁止するなら `uv run --locked ...` または先に `uv sync --locked` を使う。

出典: [Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) / [Running commands](https://docs.astral.sh/uv/concepts/projects/run/) / [Project structure and lockfile](https://docs.astral.sh/uv/concepts/projects/layout/)

## build backend

このリポジトリは C/Rust 拡張を持たない単一の pure-Python package なので、Astral が多くの Python project に推奨する `uv_build` が適合する。現状は package がリポジトリ直下の `augly_jp/` にあるため、既定の `src/` 探索を上書きする必要がある。

```toml
[build-system]
requires = ["uv_build>=0.12.13,<0.13"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "augly_jp"
module-root = ""
```

公式例は `uv_build` に uv と同じ minor 系列の上限を付けることを推奨する。ビルドスクリプトやさらに柔軟なレイアウトが必要な場合は Hatchling が代替候補だが、現状の flat layout は `module-root = ""` で公式にサポートされる。[uv build backend](https://docs.astral.sh/uv/concepts/build-backend/) / [Building distributions](https://docs.astral.sh/uv/concepts/projects/build/)

## 直接依存の最新版

「最新安定版」は調査日時点の PyPI JSON API `info.version`、Python 要件は同 API の `info.requires_python`。未宣言は互換性を保証しない。

### Runtime

| Package | 現行 | 最新安定版 | Requires-Python |
|---|---:|---:|---|
| [numpy](https://pypi.org/pypi/numpy/json) | `>=1.19.2,<1.20.0` | 2.5.3 | `>=3.12` |
| [augly](https://pypi.org/pypi/augly/json) | `^0.1.7` | 1.0.0 | `>=3.6` |
| [chikkarpy](https://pypi.org/pypi/chikkarpy/json) | `^0.1.0` | 0.1.1 | 未宣言 |
| [gensim](https://pypi.org/pypi/gensim/json) | `^4.0.1` | 4.4.0 | `>=3.9` |
| [tqdm](https://pypi.org/pypi/tqdm/json) | `^4.62.2` | 4.70.1 | `>=3.8` |
| [python-Levenshtein](https://pypi.org/pypi/python-Levenshtein/json) | `^0.12.2` | 0.27.5 | `>=3.10` |
| [ginza](https://pypi.org/pypi/ginza/json) | `^5.0.1` | 5.2.1 | `>=3.8` |
| [ja-ginza-electra](https://pypi.org/pypi/ja-ginza-electra/json) | `^5.0.0` | 5.2.0 | 未宣言 |
| [transformers](https://pypi.org/pypi/transformers/json) | `<4.10.0` | 5.17.0 | `>=3.10.0` |
| [fugashi](https://pypi.org/pypi/fugashi/json) | `^1.1.1` | 1.5.2 | `>=3.9` |
| [python-magic-bin](https://pypi.org/pypi/python-magic-bin/json) | `^0.4.14` | 0.4.14 | 未宣言 |
| [tenacity](https://pypi.org/pypi/tenacity/json) | `^8.0.1` | 9.1.4 | `>=3.10` |
| [dartsclone](https://pypi.org/pypi/dartsclone/json) | `^0.9.0` | 0.10.2 | 未宣言 |
| [torch](https://pypi.org/pypi/torch/json) | `^1.9.0` | 2.14.0 | `>=3.10` |
| [sentencepiece](https://pypi.org/pypi/sentencepiece/json) | `^0.1.96` | 0.2.2 | `>=3.9` |

`ja-ginza-electra 5.2.0` は `ginza>=5.2.0,<5.3.0` を要求するため、GiNZA の最新 5.2.1 とは整合する。`transformers` だけは上記の transitive 上限に従い 4.25.1 を採用する必要がある。

### Development

| Package | 現行 | 最新安定版 | Requires-Python |
|---|---:|---:|---|
| [pysen](https://pypi.org/pypi/pysen/json) | `^0.9.1` | 0.12.1 | `>=3.10` |
| [absl-py](https://pypi.org/pypi/absl-py/json) | `>=0.9,<0.13` | 2.5.0 | `>=3.10` |
| [pytest](https://pypi.org/pypi/pytest/json) | `^6.2.4` | 9.1.1 | `>=3.10` |
| [pytest-xdist](https://pypi.org/pypi/pytest-xdist/json) | `^2.3.0` | 3.8.0 | `>=3.9` |
| [taskipy](https://pypi.org/pypi/taskipy/json) | `^1.8.1` | 1.14.1 | `>=3.6,<4.0` |
| [jupyterlab](https://pypi.org/pypi/jupyterlab/json) | `^3.1.7` | 4.6.3 | `>=3.10` |
| [ipywidgets](https://pypi.org/pypi/ipywidgets/json) | `^7.6.4` | 8.1.9 | `>=3.7` |
| [pytest-cov](https://pypi.org/pypi/pytest-cov/json) | `^2.12.1` | 7.1.0 | `>=3.9` |
| [pytest-mock](https://pypi.org/pypi/pytest-mock/json) | `^3.6.1` | 3.15.1 | `>=3.9` |
| [pytest-asyncio](https://pypi.org/pypi/pytest-asyncio/json) | `^0.15.1` | 1.4.0 | `>=3.10` |
| [scikit-learn](https://pypi.org/pypi/scikit-learn/json) | `^0.24.2` | 1.9.1 | `>=3.11` |
| [pandas](https://pypi.org/pypi/pandas/json) | `^1.3.2` | 3.0.5 | `>=3.11` |
| [tensorboard](https://pypi.org/pypi/tensorboard/json) | `^2.6.0` | 2.21.0 | `>=3.9` |

最新 pytest 系の declared constraints は互換である。`pytest-asyncio 1.4.0` は `pytest>=8.4,<10`、pytest-xdist 3.8.0 は `pytest>=7`、pytest-cov 7.1.0 は `pytest>=7`、pytest-mock 3.15.1 は `pytest>=6.2.5` を要求する。[pytest-asyncio metadata](https://pypi.org/pypi/pytest-asyncio/1.4.0/json) / [pytest-xdist metadata](https://pypi.org/pypi/pytest-xdist/3.8.0/json) / [pytest-cov metadata](https://pypi.org/pypi/pytest-cov/7.1.0/json) / [pytest-mock metadata](https://pypi.org/pypi/pytest-mock/3.15.1/json)

## ty への型チェック移行

調査日時点の最新安定版は `ty 0.0.80`（Requires-Python `>=3.8`、Beta）。Astral の推奨どおり project の開発依存へ追加し、uv が同期した環境内で実行する。

```shell
uv add --dev ty
uv run ty check
```

バージョンを lock 内で更新する場合は `uv lock --upgrade-package ty`。ty は project の `.venv` の `site-packages` を使って外部依存を解決し、引数なしの `ty check` は project 内の Python ファイルを検査する。[ty installation](https://docs.astral.sh/ty/installation/) / [Type checking](https://docs.astral.sh/ty/type-checking/) / [ty PyPI metadata](https://pypi.org/pypi/ty/json)

新しい `uv check` も環境を同期して ty を実行し、ty が dev dependency なら `uv.lock` の版を使う。ただし現時点では uv の `check-command` preview feature なので、通常の project task / CI には安定した `uv run ty check` を優先する。[uv check CLI](https://docs.astral.sh/uv/reference/cli/#uv-check) / [uv preview features](https://docs.astral.sh/uv/concepts/preview/)

型注釈のない外部コードは即座に除外されるのではなく、ty が不明な部分を `Unknown` として扱い、得られる範囲の検査を継続する。まず suppression なしで実行するのが妥当。どうしても外部 module の解決・型情報が問題になる場合だけ、対象を限定して設定する。

```toml
[tool.ty.analysis]
# module 自体を解決できない場合に unresolved-import だけを抑止する
allowed-unresolved-imports = ["legacy_package.**"]

# 解決できても型情報を利用せず Any として扱う最終手段
replace-imports-with-any = ["untyped_package.**"]

[tool.ty.rules]
# 全 unresolved import の severity を下げたい場合（通常は対象別設定を優先）
unresolved-import = "warn"
```

`allowed-unresolved-imports` は `unresolved-import` 診断だけを抑止する。一方 `replace-imports-with-any` は解決可能な module も `Any` に置き換え、該当 import 診断を無条件に抑止するため、より広い型安全性を失う。ty は warning でも既定で exit code 1 を返す点にも注意する。[ty configuration](https://docs.astral.sh/ty/reference/configuration/) / [Typing FAQ: Unknown and imports](https://docs.astral.sh/ty/reference/typing-faq/)
