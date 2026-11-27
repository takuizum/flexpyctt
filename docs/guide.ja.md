# ユーザーガイド

[English](guide.md) | 日本語

flexpyctt は、推定済みの IRT モデルを使わない項目・テスト分析のパッケージである。このガイドでは、シミュレーションした
データを使って、各関数の使い方と結果の読み方を順に説明する。引数と戻り値の一覧は [API リファレンス](reference.ja.md) に
ある。IRT モデルの推定と、推定済みのモデルを使う分析（局所依存、モデルに基づく平行分析、次元性のレポートなど）は
[flexpyirt](https://takuizum.github.io/flexpyirt/ja/) で行う。

## 1. 準備

データは、1 行が回答者 1 人、1 列が 1 項目の DataFrame（または配列）で渡す。反応は順序のあるカテゴリのコードで、欠測は
NaN とする。乱数を使う関数は `rng`（整数のシードまたは `numpy.random.Generator`）を受け取るので、結果は再現できる。
JAX は使わず、64 ビットの設定も要らない。

以下の節では、1 つの因子を測る 2 値項目 12 個に 1500 人が回答したデータを使う。

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
items = [f"q{j:02d}" for j in range(1, 13)]
theta = rng.normal(size=1500)
slopes = rng.uniform(0.8, 1.8, size=12)
difficulties = np.linspace(-1.5, 1.5, 12)
prob = 1 / (1 + np.exp(-slopes * (theta[:, None] - difficulties)))
df = pd.DataFrame((rng.random((1500, 12)) < prob).astype(int), columns=items)
print(df.head())
```

## 2. ポリコリック相関

`flexpyctt.polychoric(data)` は順序反応のポリコリック相関行列（2 値項目ではテトラコリック相関）を推定する。固有値に
基づく次元性の確認で入力として使うのが一般的である。two-step の最尤法（Olsson, 1979）で、各項目の周辺比率から閾値を
求め、そのうえで項目のペアごとに相関を 1 つずつ求める。

```python
from flexpyctt import polychoric

pc = polychoric(df)
print(pc.correlation.iloc[:4, :4].round(3))
print(pc.thresholds[items[0]].round(3))
print(pc.n_obs.iloc[:2, :2])
```

- 回答は順序のある符号であればどんな番号付けでもよい（`1..5`、`0, 10, 20` など）。意味を持つのは順序だけである。
  欠測（NaN）は利用可能なケースで扱う。閾値は項目へのすべての回答から、各ペアは両方に回答した回答者から求める。
- 空のセルがちょうど 1 つの 2 x 2 表では、尤度は $|\rho| = 1$ で最大になる。既定では lavaan と同じく、周辺を
  変えないように空のセルを含む対角へ 0.5 を移す（`zero_cell_correction=0` で無効）。それでも $|\rho| \to 1$ で最大になる
  ペアは `HeywoodWarning` を出し、`pc.at_boundary` に示す。
- ペアごとの推定値を並べた行列は正定値とは限らない。その場合は平滑化し（psych の `cor.smooth` と同じ規則）、
  `NotPositiveDefiniteWarning` を出す。`pc.unsmoothed_correlation` にはペアごとの推定値がそのまま残る。
- `flexpyirt.polychoric()` は同じ推定量の JAX による実装で、引数と結果は同じである。2 つは $10^{-9}$ 程度で一致する。

## 3. 次元数

1 次元モデルを当てはめる前に、回答がいくつの次元を支持するかを確かめておくとよい。以下の手法はいずれも
偏りが知られた診断なので、いくつかを合わせて見る。

```python
from flexpyctt import efa, empirical_kaiser, hull, parallel_analysis, velicer_map

pa = parallel_analysis(df, rng=0, n_replications=20)
print(pa.n_factors, pa.observed[:3].round(3), pa.reference[:3].round(3))
print(velicer_map(pc).n_factors, empirical_kaiser(pc).n_factors)
print(hull(pc, rng=4).n_factors)
print(efa(pc, n_factors=1).loadings.head(3).round(3))
```

- `parallel_analysis()`（Horn, 1965）は、ポリコリック相関行列の固有値のうち、共通因子をもたない参照データの固有値の
  95 パーセンタイルを上回るものを先頭から数える。既定の参照データは各項目の回答を独立に並べ替えたもので、項目の
  周辺分布を保つ。`reference="normal"` で正規データ、`eigen_type="smc"` で縮約行列を使う。結果は `rng` によって
  変わるので、最終的な分析では反復を増やす（既定は 100）。
- `velicer_map()` は $m$ 個の主成分を除いたあとの平均偏相関を最小にする。既定は 4 乗を使う改訂版
  （`power=2` で元の版）。
- `hull()` は 0, 1, …, $J$ 因子の因子モデル（`efa()`、ULS または ML）を当てはめ、使う自由度に比べて因子を増やしても
  適合度がほとんど上がらなくなる直前の数を選ぶ（Lorenzo-Seva, Timmerman & Kiers, 2011）。項目が 6 つ以上必要。
- `comparison_data()`（Ruscio & Roche, 2012）は、観測された相関と項目の分布を再現する 1, 2, … 因子の母集団を
  生成し、その母集団からの標本の固有値が観測固有値に近づく限り因子を加える。
- `empirical_kaiser()` は各固有値を標本サイズに応じた参照値と比べる（Braeken & van Assen, 2017）。
- これらは次元の数を数えるもので、1 次元 IRT モデルが当てはまるかを検定するものではない。それには、
  flexpyirt でモデルを推定し、局所依存の統計量、`FitResult.m2()`、`FitResult.item_fit()` を使う
  （[flexpyirt のユーザーガイド](https://takuizum.github.io/flexpyirt/ja/guide/)）。

## 4. DETECT と DIMTEST

DETECT（Zhang & Stout, 1999）は、総得点（または残りの得点）が同じ回答者の中での項目ペアの共分散を見る。本質的に
一次元なら、この条件付き共分散は 0 に近い。複数の次元があると、項目のクラスタ内では正、クラスタ間では負になる。

```python
from flexpyctt import detect, dimtest, exploratory_detect

halves = {item: ("A" if i < len(items) // 2 else "B") for i, item in enumerate(items)}
print(detect(df, halves).indices.round(3))
explore = exploratory_detect(df, max_clusters=3, rng=3)
print(explore.table.round(3))
test = dimtest(df, rng=5)
print(test.assessment_items, round(test.statistic, 3), round(test.p_value, 3))
```

- `detect()` は与えた分割を評価する（確認的 DETECT）。`exploratory_detect()` は回答者の半分で分割を探し、両方の
  半分で報告する。前半のノイズにしか合わない分割は、検証用の DETECT が小さくなって分かる。
- よく使われる読み方: DETECT が 0.2 未満なら本質的に一次元、0.2〜0.4 で弱い、0.4〜1 で中程度、1 を超えると強い
  多次元性。ASSI が 0.25、RATIO が 0.36 を超えると、ほぼ単純構造であることを示す。1 次元のデータでは、この例の
  ように、どの分割でも DETECT は 0 に近いか負になる。
- DETECT には欠測のないデータが要る。flexpyirt の `dimensionality()` は一次元性の確認の 1 つとして、探索的 DETECT（検証用の
  半分）を含める。
- `dimtest()`（Stout, 1987; Stout, Froelich & Gao, 2001）は同じ考え方を 2 値項目の検定にしたもので、回答者の
  3 分の 1 で検定用の部分テストを選び、残りで検定する。保守的なので、小さい $p$ は第 2 の次元の強い証拠だが、大きい
  $p$ は一次元性の弱い証拠にとどまる。

## 5. 古典的項目分析

IRT モデルを当てはめる前に（または当てはめずに）、古典的な項目統計量で問題のある項目を探せる。

```python
from flexpyctt import (
    coefficient_alpha,
    coefficient_omega,
    distractor_analysis,
    item_analysis,
    score_responses,
)

# 多肢選択の生の回答: 正答は "B"、誤答は A・C・D に散らばる。
choice_rng = np.random.default_rng(20)
wrong = choice_rng.choice(np.array(["A", "C", "D"]), size=df.shape)
raw = pd.DataFrame(np.where(df.to_numpy() == 1, "B", wrong), columns=items)
key = {item: "B" for item in items}

scores = score_responses(raw, key=key)
print(item_analysis(scores).round(3).head())
alpha = coefficient_alpha(scores)
print(round(alpha.alpha, 3), np.round(alpha.confidence_interval, 3))
print(round(coefficient_omega(scores, correlation="polychoric").omega, 3))
options = distractor_analysis(raw, key=key)
print(options[options["item"] == items[0]].round(3))
```

- `score_responses()` は選んだ選択肢を 0 / 1 の得点にする。選択肢は `==` で比べるので、キーはデータと同じ
  コードと型で書く（`1` と `"1"` は一致しない。誰も選んでいない正答選択肢があると警告が出る）。無回答は 0 点、
  `missing="keep"` なら NaN のまま。
- `item_analysis()` は項目ごとに 1 行を返す。`difficulty`（平均を最大得点で割ったもの。0 / 1 項目では正答率）、
  `item_total` と `item_rest`（合計得点との相関と、その項目を除いた合計との相関。0 / 1 項目では点双列相関）、
  `alpha_if_deleted`。見るのは `item_rest` のほう。`item_total` は項目自身を含むので、項目数が少ないと大きく出る。
  `correlation="polyserial"` にすると、2 段階推定による多分系列相関（0 / 1 項目では双列相関）を返す。
- `coefficient_alpha()` は 0 / 1 項目では KR-20 になる。標準化 α、Feldt の信頼区間、各項目を除いたときの α
  （`alpha.if_deleted`）も返す。
- `coefficient_omega()` は項目間相関に 1 因子モデルを（ULS で）当てはめ、ω total を返す。
  psych の `omega()` と同じく、負の負荷量の項目は反転する（警告を出し、`reversed_items` に挙げる）。報告する合計が
  採点したままの項目を使うなら、その項目は自分で逆転採点しておく。
  `correlation="polychoric"` では、観測された合計得点ではなく、カテゴリの背後にある潜在反応についての値になる。
- 欠測（NaN）は既定ではペアワイズに使う。共分散・相関のそれぞれを、両方の項目に答えた回答者から計算する。
  `missing="listwise"` なら欠測のない行だけを使う。
- `distractor_analysis()` は項目と選択肢の組ごとに 1 行を返す。うまく働いている誤答選択肢は、得点の高い群ほど
  選ばれにくく（`proportion_group_1` が得点の最も低い 3 分の 1）、残りの項目の合計との点双列相関
  （`point_biserial`）が負になる。正答はその逆になるはず。この例では誤答を無作為に選ばせたので、同じ項目の誤答
  選択肢どうしは似た値になる。

## 6. 観測得点による DIF

flexpyctt は IRT モデルを推定せずに DIF のある項目を探せる。回答者を合計得点（または `anchor_items` の合計、外部の
`matching` 変数）でそろえ、得点ごとに群を比べる。`mantel_haenszel()` は Mantel–Haenszel の $\chi^2$、共通オッズ比
$\hat\alpha_{MH}$、ETS のデルタ $\Delta_{MH}=-2.35\ln\hat\alpha_{MH}$ と ETS の A / B / C 分類を、`logistic_dif()` は
入れ子のロジスティック回帰で均一・不均一 DIF を検定し、Nagelkerke の $\Delta R^2$ を返す。`mantel()`（順序のある
多値項目）と `generalized_mantel_haenszel()`（複数の焦点群、順序のないカテゴリ）も同じ引数をとる。`purify=True`
では、検出された項目をマッチング得点から外し、検出される項目の組が繰り返すまで検定をやり直す。

```python
from flexpyctt import logistic_dif, mantel_haenszel

# 2 値項目 20 個。焦点群は平均が低く、項目 q03 と q07 は焦点群にとって難しい（均一 DIF）。
dif_rng = np.random.default_rng(7)
dif_theta = np.concatenate([dif_rng.normal(0.0, 1.0, 1200), dif_rng.normal(-0.5, 1.0, 800)])
dif_group = np.repeat(["ref", "focal"], [1200, 800])
dif_shift = np.where(dif_group[:, None] == "focal", np.isin(np.arange(20), [2, 6]) * 0.8, 0.0)
dif_prob = 1 / (1 + np.exp(-1.2 * (dif_theta[:, None] - np.linspace(-1.5, 1.5, 20) - dif_shift)))
dif_df = pd.DataFrame(
    (dif_rng.random((2000, 20)) < dif_prob).astype(int),
    columns=[f"q{j:02d}" for j in range(1, 21)],
)

mh = mantel_haenszel(dif_df, group=dif_group, focal="focal", purify=True, p_adjust="holm")
print(mh.table[["item", "chi2", "delta", "ets_class", "adj_p", "flagged"]].round(3).head(8))
print(mh.flagged_items, mh.n_iter, mh.converged)
lr = logistic_dif(dif_df, group=dif_group, focal="focal", purify=True, p_adjust="holm")
print(lr.table.loc[lr.table["flagged"], ["item", "chi2", "delta_r2", "jodoin_gierl_class"]])
```

- $\Delta_{MH}<0$ は、同じ合計得点で項目が焦点群にとって難しいことを表す。C 分類（ここでは q03 と q07）には
  $|\Delta_{MH}|\ge1.5$ かつ 1 より有意に大きいことが要る。A 分類は $|\Delta_{MH}|<1$ または有意でないこと。
  `classification="effect_size"` は閾値だけを使う。
- 観測した合計得点でそろえるので、群の能力差が大きく項目が少ないと、DIF のない項目が検出されることがある（ここでは
  q17。効果量は無視できる大きさ）。p 値だけでなく効果量も見る。
- マッチングに使う項目は欠測がないこと。欠測のある行は先に除く。

## 7. エラーと警告

flexpyctt が意図して送出する例外は `FlexpycttError`、警告は `FlexpycttWarning` を継承する（`flexpyctt.errors`）。

- `SpecificationError`（`ValueError` でもある）: 指定がそれ自体と、またはデータと矛盾している。範囲外のオプション、
  数値でない反応、カテゴリが 1 つしか観測されていない項目など。計算を始める前に送出する。
- `ConvergenceWarning`: 収束基準を満たさずに推定が止まった（`max_iter` に達した場合など）。
- `HeywoodWarning`: 境界解に達した（相関が 1、独自性が境界にある、など）。
- `NotPositiveDefiniteWarning`: 相関行列（ペアごとに推定したポリコリック行列など）が正定値でなかったので、平滑化した。

flexpyirt は同じクラスを `flexpyirt.errors` から再公開している。
