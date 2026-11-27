# コントリビューションガイド

[English](CONTRIBUTING.md) | 日本語

flexpyctt は関連する 3 つのパッケージの 1 つで、それぞれが自分のリポジトリとバージョンを持つ。

| パッケージ | リポジトリ | 内容 |
|---|---|---|
| `flexpyirt` | [takuizum/flexpyirt](https://github.com/takuizum/flexpyirt) | IRT モデルの推定と、推定済みのモデルを使うものすべて。JAX と flexpyctt に依存 |
| `flexpyctt` | [takuizum/flexpyctt](https://github.com/takuizum/flexpyctt) | 推定済みのモデルを使わない項目・テスト分析。NumPy・SciPy・pandas のみ |
| `flexpyequate` | [takuizum/flexpyequate](https://github.com/takuizum/flexpyequate) | 得点分布の等化。NumPy・SciPy・pandas のみ。他の 2 つとは独立 |

- **flexpyctt は JAX も flexpyirt も import しない。**CI で、どちらもインストールされていないことを確認する。
  乱数を使う関数は `rng`（シードまたは `numpy.random.Generator`）を受け取る。
- 推定済みのモデル（`FitResult`）が要る道具は flexpyirt に、データだけで済む道具はここに置く。
- これらの道具の設計の記録は、いまも flexpyirt の設計文書
  （[DESIGN.ja.md](https://github.com/takuizum/flexpyirt/blob/develop/docs/dev/DESIGN.ja.md) §16.8〜§16.16、
  §16.18、§16.20、§16.21、§16.24）にあり、照合ケース（V24、V25、V27〜V30、V46、V47）は flexpyirt の
  [VALIDATION.ja.md](https://github.com/takuizum/flexpyirt/blob/develop/docs/dev/VALIDATION.ja.md) にある。
  docstring の「DESIGN.md §…」はその文書を指す。
- flexpyirt は非公開のヘルパーを 4 つ import している: `correlation._coerce_ordinal` と
  `dimensionality._observed_correlation`、`_eigenvalues`、`_descending_eigenvalues`。これらのシグネチャと挙動は
  公開 API と同じに扱う（`tests/test_package.py` がシグネチャを確認する）。

## ブランチ

| ブランチ | 役割 | 作成元 / マージ先 |
|---|---|---|
| `main` | リリース済みの状態だけ。直接 push しない | `develop` からのリリース PR だけを受け付ける |
| `develop` | 統合ブランチ。次のリリースに入るものを集める | `feature/*` と `fix/*` からの PR を受け付ける |
| `feature/<slug>` | 機能、ドキュメント、リファクタリングなど | `develop` から作り、`develop` へ PR |
| `fix/<issue番号>-<slug>` | バグ修正。**必ず issue に紐づける** | `develop` から作り、`develop` へ PR |

- `<slug>` に使えるのは小文字、数字、`.`、`_`、`-` だけ。
- fix ブランチの PR の説明には `Closes #<issue番号>` を書き、その issue は open でなければならない
  （`develop` はデフォルトブランチではないので、issue はマージ時ではなく**リリース時**にワークフローが閉じる）。
- これらは `PR rules` チェック（`scripts/check_pr.py`）がすべての PR で確認する。

## PR とレビュー

- すべての変更は PR で入れる。`main` と `develop` はルールセットで保護されていて、CI（`test (3.12/3.13)`）と
  `pr-rules` が通るまでマージできない。
- PR はリポジトリのオーナーがレビューしてからマージする（自動マージはしない）。
- 数値計算のコードを変えるときは、R（lavaan, psych, ltm, polycor, EFAtools, EFA.dimensions, sirt, difR）との照合テストが通ること。参照値は
  `tests/reference/` にコミットしてあり、CI では R を実行しない。

## リリース

バージョンは **git タグ**（`vX.Y.Z`、hatch-vcs）から決まる。`pyproject.toml` にはバージョンを書かない。
リリースとはタグを付けることである。タグのない開発中のコミットは `0.1.1.dev3+g1a2b3c4` のようなバージョンになる。

1. `develop` から `main` へ PR を開く。**タイトルでバージョンの上げ方が決まる**:
   - `release(major): ...` / `release(minor): ...` / `release(patch): ...`
   - **マージコミット**でマージする（`main` のルールセットはマージコミットだけを許可している）。
2. マージすると `.github/workflows/release.yml` が動く:
   1. `plan` ジョブ: PR のタイトルと最新の `vX.Y.Z` タグから次のバージョンを計算する。
   2. `publish` ジョブ: **`release` 環境での承認を待つ**。オーナーが Actions のページで承認すると、`main` に
      `vX.Y.Z` のタグを付け、wheel と sdist をビルドし（`uv build`）、GitHub Release を作る。そのあと、
      このリリースに含まれる fix PR が `Closes` した issue を閉じる。
   3. `pypi` ジョブ: **同じ wheel と sdist を PyPI に公開する**（`pypi` 環境、Trusted Publishing。API トークンは
      使わない）。
3. **最初のリリース**（`v*` タグがまだないとき）は、タイトルの指定にかかわらず **v0.1.0** になる。
4. 手動実行（`workflow_dispatch`）はドライランだけで、次のバージョンの計算とビルドをする。

バージョンは [Semantic Versioning](https://semver.org/) に従う。1.0.0 より前は、マイナーリリース（`0.X.0`）で
公開 API を変えてよく、パッチリリース（`0.x.Y`）では変えてはならない。flexpyirt は
`flexpyctt>=0.X,<0.(X+1)` を要求し、flexpyctt の公開名を再公開しているので、ここで互換性を壊す変更をするには
マイナーリリースと、それに合わせた flexpyirt のリリースが要る。

## セキュリティ（誰がリリースできるか）

- **`main` を更新できるのはリポジトリの管理者だけ**（ルールセット `main-release-gate`）。PR と CI の要件
  （ルールセット `main`）は管理者にも適用される。
- **`publish` ジョブは `release` 環境で動き**、この環境は `main` からしかデプロイしない。
- **PyPI へのアップロードは Trusted Publishing**（OIDC）。PyPI のプロジェクト `flexpyctt` は、
  「リポジトリ `takuizum/flexpyctt`、ワークフロー `release.yml`、環境 `pypi`」という発行者だけを信頼するので、
  リポジトリに API トークンは置かない。`pypi` 環境はオーナーの承認を必要とする。PyPI に一度アップロードした
  バージョン番号は、削除しても再利用できない。
- ワークフローの `GITHUB_TOKEN` は既定で読み取り専用で、書き込むのは `publish` ジョブだけである。

## ローカル開発

Python 環境は `uv` だけで管理する（pip は使わない）。

```bash
uv sync --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

参照値を書き出す R スクリプトは `validation/r/` にある（リポジトリのルートから実行する。例:
`Rscript validation/r/m5_polychoric_reference.R`）。出力先は `tests/reference/` である。
`validation/simulation/` には DIMTEST のシミュレーション研究がある。

## 命名

名前は PEP 8 に従い、R ではなく Python として読めるものにする。`ruff` が `N` ルールを確認する。

- 公開名は略さず、同じ概念には API 全体で同じ名前を使う（`n_factors`、`n_categories`）。R の引数名を
  持ち込まず、R での名前は docstring に書く。
- `Literal` の選択肢は小文字の `snake_case`。定着した頭字語は大文字のままにする。
- 公開関数・クラスには型ヒントと NumPy スタイルの docstring を付ける。数式を実装した関数は、docstring に
  その式と出典を書く。

## ドキュメント

どの文書にも英語版（`*.md`）と日本語版（`*.ja.md`）があり、内容は同じである。片方を変えたら両方を直す。
`README.md` / `README.ja.md` は PyPI のプロジェクトページにも表示されるので、リンクは絶対 URL にし、
開発に関する情報は書かない。
