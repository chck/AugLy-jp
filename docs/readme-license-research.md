# README badge and license research

Checked on 2026-09-14. Sources below are first-party documentation, APIs, or
repository metadata. HTTP 200 alone was not treated as success for SVG badges;
their rendered text was also inspected.

## Badge findings

| Existing badge | Finding | Recommended image URL | Recommended target |
| --- | --- | --- | --- |
| PyPI version | Works and reports `2021.9.30`, matching both the current PyPI release and `pyproject.toml`. Badge Fury also works, but Shields keeps the badge set consistent. | `https://img.shields.io/pypi/v/augly-jp.svg` | `https://pypi.org/project/augly-jp/` |
| Python versions | The endpoint works, but reports `3.8 \| 3.9` from the last published distribution. That is accurate for PyPI but stale for this checkout, whose `requires-python` is `>=3.11,<3.12`. Until a new release updates PyPI metadata, use a static `3.11` badge if the badge is meant to describe the repository. | `https://img.shields.io/badge/python-3.11-blue.svg` now; `https://img.shields.io/pypi/pyversions/augly-jp.svg` after publishing compatible metadata | `https://www.python.org/downloads/release/python-31115/` now; PyPI project page after release |
| GitHub Actions | The legacy `/workflows/Test/badge.svg` image still renders, but GitHub's documented form identifies the workflow file. The current link also spells the repository `Augly-jp` and uses the legacy query UI. | `https://github.com/chck/AugLy-jp/actions/workflows/test.yml/badge.svg?branch=main` | `https://github.com/chck/AugLy-jp/actions/workflows/test.yml` |
| Codecov | The Shields proxy works and reports 83%. Codecov documents using its own repository badge; the direct badge also rendered 83% and avoids an extra badge service. | `https://codecov.io/gh/chck/AugLy-jp/graph/badge.svg` | `https://app.codecov.io/gh/chck/AugLy-jp` |
| LGTM quality | Broken. The SVG body says `404: badge not found`, and the target redirects to GitHub's LGTM retirement announcement. Remove it. No CodeQL workflow exists in this repository, so replacing it with a security-analysis badge would be misleading. | Remove | Remove |
| Black style | The static image works, but is stale because `pyproject.toml` now runs Ruff formatting/linting and ty, not Black. Replace it with a Ruff tool badge or omit it; this is a tool label, not a quality signal. | `https://img.shields.io/badge/lint%20%26%20format-Ruff-261230.svg` | `https://docs.astral.sh/ruff/` |
| PyPI downloads | Not currently present. Shields supports monthly PyPI downloads; the live endpoint rendered `22/month`. This is optional because low-volume counts can add noise. | `https://img.shields.io/pypi/dm/augly-jp.svg` | `https://pypi.org/project/augly-jp/` |
| License | Not currently present as a badge. The live PyPI badge renders `MIT`, consistent with the published metadata, `pyproject.toml`, and `LICENSE`. | `https://img.shields.io/pypi/l/augly-jp.svg` | `https://github.com/chck/AugLy-jp/blob/main/LICENSE` |

Primary sources:

- [GitHub: adding a workflow status badge](https://docs.github.com/en/actions/how-tos/monitor-workflows/add-a-status-badge)
- [Repository workflow API](https://api.github.com/repos/chck/AugLy-jp/actions/workflows/test.yml)
- [Codecov: status badges](https://docs.codecov.com/docs/status-badges)
- [Codecov Action: OIDC authentication](https://github.com/codecov/codecov-action#using-oidc)
- [Shields: PyPI downloads badge](https://shields.io/badges/py-pi-downloads)
- [Shields: PyPI license badge](https://shields.io/badges/py-pi-license)
- [PyPI project metadata API](https://pypi.org/pypi/augly-jp/json)
- [GitHub: LGTM retirement](https://github.blog/news-insights/product-news/the-next-step-for-lgtm-com-github-code-scanning/)

The latest main-branch workflow found `coverage.xml`, but Codecov rejected the
upload because the protected branch required authentication. The action did not
fail the job because `fail_ci_if_error` defaults to false. Codecov's official
action supports token-free OIDC authentication with `use_oidc: true` when the
workflow grants `id-token: write`; enabling `fail_ci_if_error` also prevents a
future upload error from leaving a falsely green coverage step.

## License findings

### Facts

- This repository's `LICENSE`, GitHub license detection, `pyproject.toml`, and
  published PyPI metadata all identify MIT. Git history shows the MIT file was
  originally added from AugLy and then changed to this project's copyright
  holder.
- Current upstream [Meta AugLy is MIT](https://github.com/facebookresearch/AugLy/blob/main/LICENSE).
  The other projects listed under README “Inspired” are also MIT:
  [nlpaug](https://github.com/makcedward/nlpaug/blob/master/LICENSE) and
  [TextAttack](https://github.com/QData/TextAttack/blob/master/LICENSE).
- The package imports AugLy and installs its libraries as dependencies; no
  vendored third-party license or source tree was found in the tracked files.
  A dependency's license does not by itself describe the license of this
  project's distribution. If third-party code or data is copied into a future
  sdist/wheel, its terms and notices must be handled separately.
- README's current Apache-2.0 statement links to the *training code* license for
  `cl-tohoku/bert-japanese`. The model fetched at runtime is a different
  artifact: its [official Hugging Face model card](https://huggingface.co/tohoku-nlp/bert-base-japanese-v2)
  currently has inconsistent declarations (front matter says CC-BY-SA-4.0;
  prose says CC-BY-SA-3.0). The repository does not bundle that model. Therefore
  the README sentence is not evidence that AugLy-jp itself should be Apache-2.0.
- Modern Python packaging represents the distribution license as an SPDX
  expression and can explicitly include license files. PyPA's
  [license guidance](https://packaging.python.org/en/latest/guides/licensing-examples-and-user-scenarios/)
  recommends `license = "MIT"` and `license-files = ["LICENSE"]` for an existing
  MIT package, unless the built distribution incorporates material under other
  terms.

### Recommendation (not legal advice)

Keep MIT. It is consistent with the project's established license and current
upstream AugLy, and the repository inspection found no bundled work requiring a
different whole-project license. Do not switch the project to Apache-2.0 merely
because runtime models or dependencies have other licenses.

Separately, remove or rewrite the README claim that the software “includes” the
Apache-2.0 BERT work. A short third-party/model notice should instead say that
optional runtime models retain their own terms and link to each model card. Add
`license-files = ["LICENSE"]` to `[project]` so the built distribution records
the MIT license file explicitly. If code was copied rather than merely imported
from an upstream project, perform a provenance review and preserve the relevant
copyright/license notices before release.
