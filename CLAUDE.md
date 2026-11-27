# CLAUDE.md — flexpyctt

**flexpyctt** は、推定済みの IRT モデルを使わない項目・テスト分析（ポリコリック相関、次元性の診断、古典的項目分析、
観測得点による DIF）のライブラリです。NumPy・SciPy・pandas だけに依存します（JAX は使わない）。2026-11-27 に
[takuizum/flexpyirt](https://github.com/takuizum/flexpyirt) の `packages/flexpyctt/` から独立したリポジトリになりました。

各文書は英語版 `*.md` と日本語版 `*.ja.md` があり、内容は同じ。更新するときは両方を直す。

## 関連リポジトリ

| パッケージ | リポジトリ | 関係 |
|---|---|---|
| `flexpyirt` | takuizum/flexpyirt | flexpyctt に依存し（`flexpyctt>=0.X,<0.(X+1)`）、flexpyctt の公開名を再公開する |
| `flexpyequate` | takuizum/flexpyequate | 独立（どちらからも依存しない） |

- 設計は `docs/DESIGN.md`（各道具の数式・決定・照合ケース V24、V25、V27〜V30、V46、V47）。作業の前に読み、迷ったら
  先にそこで決めてからコードを書く。文書の中の「flexpyirt §…」は flexpyirt の `docs/dev/DESIGN.md` を指す
  （もとの計画 §16.8、JAX 版ポリコリック §16.9、分割の決定 §16.18・§16.24）。
- 利用者向けのドキュメントも flexpyirt のサイト（<https://takuizum.github.io/flexpyirt/>）にある。

## 最優先の原則

1. **正しさ > 速さ > 機能の多さ。** 各機能は R（lavaan・psych・ltm・polycor・EFAtools・EFA.dimensions・sirt・difR）と
   数値照合するテストが通るまで「完成」としない。R の焼き直しではなく数式から実装し、R は検証の参照値としてのみ使う。
2. **機能を水増ししない。** 未検証の機能を README に載せない。
3. **JAX も flexpyirt も import しない。** 乱数は `rng`（`numpy.random.Generator` または整数のシード）で受け取る。
   `FitResult` が要る道具は flexpyirt に置く。
4. **公開 API の互換性。** flexpyirt がマイナーバージョンの範囲で依存しているので、パッチリリース（`0.x.Y`）で公開 API を
   壊さない。壊す変更はマイナーリリースにし、flexpyirt 側の依存範囲の更新とセットで行う。flexpyirt が import している
   非公開ヘルパー 4 つ（`_coerce_ordinal`、`_observed_correlation`、`_eigenvalues`、`_descending_eigenvalues`）も
   同じ扱いにする。
5. **小さくコミット。**

## 開発環境

- Python >= 3.12、パッケージ管理は `uv` のみ（pip は使わない）。
- 依存: `numpy`, `scipy`, `pandas`。開発依存: `pytest`, `pytest-xdist`, `ruff`, `mypy`。
- 検査: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`
- R 参照値の生成: `validation/r/*.R`（リポジトリのルートから実行）。生成した参照値は `tests/reference/*.json` に
  コミットし、CI では R を走らせない。

## コーディング規約

- 公開関数・クラスは型ヒント必須、docstring は NumPy スタイル。公開 API は keyword-only 引数を基本にする。
- 数式を実装した関数には、docstring に対応する式（LaTeX）と出典（著者・年）を書く。
- 例外は `flexpyctt.errors` に定義した具体的なクラスで投げる。
- 命名は PEP 8 に従い、R の引数名を持ち込まない（CONTRIBUTING.md「命名」）。

## 開発フロー（CONTRIBUTING.md）

- `feature/<slug>` と `fix/<issue番号>-<slug>` を `develop` から切り、`develop` へ PR。fix は issue 必須（`Closes #N`）。
- リリースは `develop` → `main` の PR。タイトル `release(major|minor|patch): ...` で上げ方を決め、マージするとすぐに
  タグ `vX.Y.Z` と GitHub Release ができ（承認なし）、`pypi` ジョブはオーナーが `pypi` 環境を承認したあとに PyPI に
  公開する（Trusted Publishing）。バージョンは git タグから（hatch-vcs、`pyproject.toml` に書かない）。最初のリリースは v0.1.0。
- `README*.md` は PyPI にも表示されるのでリンクは絶対 URL。開発関連の情報は書かない。
- **PR はオーナーがレビューしてマージする。Claude は PR を作るまでで、マージしない。**
