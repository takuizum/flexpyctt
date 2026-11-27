# API リファレンス

[English](reference.md) | 日本語

flexpyctt の公開 API（`from flexpyctt import ...` で使えるものすべて）。使い方の流れは
[ユーザーガイド](guide.ja.md) を参照。docstring にも同じ情報と、数式・出典がある（`help(flexpyctt.polychoric)` など）。

最初の 1〜2 個を除き、公開関数の引数はキーワード専用である（`*` の後ろ）。乱数を使う関数は `rng`（整数のシードまたは
`numpy.random.Generator`）を受け取る。[flexpyirt](https://takuizum.github.io/flexpyirt/ja/) はここにある名前を同じ名前で再公開する。ただし
`polychoric` は別で、`flexpyirt.polychoric` は同じ推定量の JAX による実装である。

- [`polychoric`](#polychoric), [`PolychoricResult`](#polychoricresult), [`parallel_analysis`](#parallel_analysis), [`ParallelAnalysisResult`](#parallelanalysisresult), [`velicer_map`](#velicer_map), [`MAPResult`](#mapresult), [`empirical_kaiser`](#empirical_kaiser), [`EmpiricalKaiserResult`](#empiricalkaiserresult), [`efa`](#efa), [`EFAResult`](#efaresult), [`hull`](#hull), [`HullResult`](#hullresult), [`comparison_data`](#comparison_data), [`ComparisonDataResult`](#comparisondataresult), [`detect`](#detect), [`DetectResult`](#detectresult), [`exploratory_detect`](#exploratory_detect), [`ExploratoryDetectResult`](#exploratorydetectresult), [`dimtest`](#dimtest), [`DimtestResult`](#dimtestresult), [`score_responses`](#score_responses), [`item_analysis`](#item_analysis), [`coefficient_alpha`](#coefficient_alpha), [`AlphaResult`](#alpharesult), [`coefficient_omega`](#coefficient_omega), [`OmegaResult`](#omegaresult), [`distractor_analysis`](#distractor_analysis), [`mantel_haenszel`](#mantel_haenszel), [`mantel`](#mantel), [`generalized_mantel_haenszel`](#generalized_mantel_haenszel), [`logistic_dif`](#logistic_dif), [`DIFResult`](#difresult), [`flexpyctt.errors`](#flexpyctterrors)

---

## 相関

インポート: `from flexpyctt import polychoric`。

### `polychoric`

```text
polychoric(data, *, smooth=True, zero_cell_correction=0.5, tol=1e-10,
           max_iter=100) -> PolychoricResult
```

順序反応のポリコリック相関行列（2 値項目ではテトラコリック相関）を two-step の最尤法で求める
（[ガイド 2 節](guide.ja.md#2-ポリコリック相関)）。各項目の閾値を周辺比率から求め、そのうえで各ペアの相関を、離散化した
2 変量正規分布のもとでのペアの分割表の尤度を最大にする値とする（Olsson, 1979）。2 変量正規分布の確率は Owen (1956) の
$T$ 関数で計算する。順序変数に対する lavaan の `lavCor` と一致する。

`flexpyirt.polychoric` は引数と結果が同じで、同じ推定量を JAX で計算する（2 変量正規分布の確率は Genz, 2004 の方法）。
JAX の 64 ビットが無効なときは `PrecisionError` を出す。2 つは $10^{-9}$ 程度で一致する。

| 引数 | 型 | 説明 |
|---|---|---|
| `data` | DataFrame / 2 次元配列 | 1 行が 1 回答者、1 列が 1 項目。順序のあるカテゴリの符号（任意の数値で、昇順がカテゴリの順序）、欠測は NaN。各項目に観測されたカテゴリが 2 つ以上必要。観測されない符号はカテゴリとみなさない |
| `smooth` | bool | 正定値でない行列を平滑化する（psych の `cor.smooth` と同じ規則）。`NotPositiveDefiniteWarning` を出す（既定 `True`） |
| `zero_cell_correction` | float | 空のセルがちょうど 1 つの 2 x 2 表で、空のセルを含む対角にこの値を足し、もう一方の対角から引く（周辺は変わらない。lavaan の既定の規則）。補正しないと推定値は $\pm 1$ になる。`0` で無効（既定 `0.5`） |
| `tol` | float | 1 観測あたりの勾配に対する収束判定（既定 `1e-10`） |
| `max_iter` | int | ペアごとの Newton 反復の上限（既定 `100`） |

欠測は利用可能なケースで扱う。閾値は項目への観測された回答をすべて使い、ペアの分割表は両方に回答した回答者を使う。

**戻り値**: [`PolychoricResult`](#polychoricresult)

**例外**: `SpecificationError`（数値でないデータ、項目が 2 つ未満、観測された
カテゴリが 1 つだけの項目、両方に回答した回答者がいないペア、またはその回答者の中で一方の項目が一定）。

**警告**: `HeywoodWarning`（ペアの尤度が $|\rho| \to 1$ で最大）、`ConvergenceWarning`（収束しないペアがある）、
`NotPositiveDefiniteWarning`（行列を平滑化した）。

### `PolychoricResult`

`polychoric()` の戻り値。変更不可。

| 属性 | 型 | 説明 |
|---|---|---|
| `correlation` | DataFrame | 項目 x 項目の相関行列（`smooth=True` で必要なときは平滑化後） |
| `unsmoothed_correlation` | DataFrame | 平滑化前のペアごとの推定値 |
| `thresholds` | dict[str, ndarray] | 項目ごとに、観測された $K$ カテゴリの $K - 1$ 個の閾値 $\tau_k = \Phi^{-1}(P(Y \le k))$ |
| `categories` | dict[str, ndarray] | 項目ごとに、観測された回答の値（昇順） |
| `n_obs` | DataFrame | 両方の回答が観測された回答者数（対角は項目への回答数） |
| `converged`, `at_boundary`, `zero_cell_corrected` | DataFrame (bool) | ペアごとに、収束したか、$\lvert\rho\rvert \to 1$ で最大か（推定値は $\pm\tanh 8$）、`zero_cell_correction` を適用したか |
| `smoothed` | bool | 平滑化で行列が変わったか |
| `min_eigenvalue` | float | `unsmoothed_correlation` の最小固有値 |

## 次元数

いずれも「次元はいくつか」を診断として答える。1 つの手法だけを信頼せず、いくつかを組み合わせて判断する
（[ガイド 3 節](guide.ja.md#3-次元数)）。

推定済みのモデルを使う道具（`model_parallel_analysis`、`dimensionality`、局所依存の統計量）は
[flexpyirt](https://takuizum.github.io/flexpyirt/ja/reference/) にある。

### `parallel_analysis`

```text
parallel_analysis(data, *, rng, correlation="polychoric", eigen_type="pca",
                  reference="permutation", n_replications=100, quantile=0.95,
                  zero_cell_correction=0.5) -> ParallelAnalysisResult
```

Horn (1965) の平行分析。観測された相関行列の固有値のうち、共通因子をもたない参照データの固有値の `quantile`
（Glorfeld, 1995）を上回るものを先頭から数える。

| 引数 | 型 | 説明 |
|---|---|---|
| `data` | DataFrame / 2 次元配列 | 回答者 x 項目（順序のある符号。`correlation="pearson"` なら連続量でもよい）。欠測は NaN |
| `rng` | int / `numpy.random.Generator` | 参照データのシードまたは生成器（例 `rng=0`） |
| `correlation` | `"polychoric"` / `"pearson"` | データと並べ替えの反復で使う相関（正定値でなければどちらも平滑化） |
| `eigen_type` | `"pca"` / `"smc"` | 相関行列の固有値、または対角を重相関係数の 2 乗にした縮約行列の固有値 |
| `reference` | `"permutation"` / `"normal"` | 項目ごとに独立に並べ替える（周辺分布を保つ。Garrido, Abad & Ponsoda, 2013）、または同じ大きさの独立な正規データ（Pearson） |
| `n_replications` | int | 参照データの数（既定 `100`） |
| `quantile` | float / None | 参照固有値の (0, 1) の分位点（既定 `0.95`）。`None` で平均 |
| `zero_cell_correction` | float | `polychoric()` に渡す |

**戻り値**: [`ParallelAnalysisResult`](#parallelanalysisresult)

**例外**: `SpecificationError`（不正な指定やデータ、項目が 3 つ未満）。

**警告**: 並べ替えの反復を飛ばしたとき（欠測のため、並べ替えたペアに使える回答がない）に `FlexpyirtWarning`。

### `ParallelAnalysisResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | `reference` を上回る観測固有値の数（先頭から） |
| `observed` | ndarray | 観測固有値（降順） |
| `reference`, `reference_mean` | ndarray | 位置ごとの参照の分位点（または平均）と平均 |
| `replications` | ndarray | `(n_replications - n_skipped, p)` の参照固有値 |
| `correlation` | DataFrame | 観測された相関行列 |
| `eigen_type`, `reference_kind`, `correlation_kind`, `quantile`, `n_skipped` | | 使った設定と飛ばした反復の数 |

### `velicer_map`

```text
velicer_map(correlation, *, power=4) -> MAPResult
```

Velicer の最小平均偏相関（MAP）検定。最初の $m$ 個の主成分を偏相関で除いたあと、非対角の偏相関の `power` 乗の
平均を $m = 0, \dots, p - 1$ で最小にする。`power=2` は元の検定（Velicer, 1976）、`power=4` は改訂版（Velicer, Eaton &
Fava, 2000）。`correlation` は行列（DataFrame / 配列）または `PolychoricResult`。

### `MAPResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | 最小にする $m$ |
| `statistic` | ndarray | $m = 0, \dots, p - 1$ の平均偏相関の統計量（偏分散が 0 になって以降は NaN） |
| `power` | int | 2 または 4 |

### `empirical_kaiser`

```text
empirical_kaiser(correlation, *, n_obs=None) -> EmpiricalKaiserResult
```

経験的カイザー基準（Braeken & van Assen, 2017）。$j$ 番目の固有値を
$\max\{(1+\sqrt{p/n})^2 (p-\sum_{i<j}\lambda_i)/(p-j+1),\ 1\}$ と比べる。`correlation` が `PolychoricResult` でなければ
`n_obs` が必要（`PolychoricResult` ならペアごとの $n$ の最小値を使う）。

### `EmpiricalKaiserResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | 参照値を上回る固有値の数（先頭から） |
| `eigenvalues`, `reference` | ndarray | 固有値と参照値 |
| `n_obs` | int | 使った標本サイズ |

### `efa`

```text
efa(correlation, *, n_factors, method="uls", n_obs=None) -> EFAResult
```

相関行列（DataFrame、配列、`PolychoricResult`）の回転しない探索的因子分析。ULS（minres。Harman & Jones, 1966）または
ML（Jöreskog, 1967）で、独自性を $1 - \mathrm{SMC}$ から $[0.005, \max(1, \mathrm{SMC})]$ の範囲で最小化する。psych の
`fa(rotate = "none")`、EFAtools の `EFA(rotation = "none")` と同じ。`n_obs`（`PolychoricResult` ならペアごとの $n$ の
最小値が既定）を与えると、適合度に Bartlett の補正をした $\chi^2$、その $p$ 値、CFI、RMSEA が加わる。CAF と $df$ は
常に報告する。独自性が境界に達すると `HeywoodWarning`、最適化が途中で止まると `ConvergenceWarning` を出す。

### `EFAResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `loadings` | DataFrame | 回転しない負荷量（列の和が非負になるよう符号をそろえる） |
| `uniquenesses`, `communalities` | Series | $\psi_i$ と $h_i^2$ |
| `fit` | dict | `caf`、`df`、`n_obs` があれば `chi2`、`p`、`cfi`、`rmsea` |
| `method`, `converged`, `heywood`, `objective` | | 推定法と診断 |

### `hull`

```text
hull(correlation, *, n_obs=None, max_factors=None, rng=None, method="uls",
     fit_index="caf", n_replications=1000) -> HullResult
```

Hull 法（Lorenzo-Seva, Timmerman & Kiers, 2011）。EFAtools の `HULL()` と同じ。$k = 0, \dots, J$ 因子の解を適合度
（CAF、CFI、$1-$RMSEA）と自由度で並べ、上側の凸包の上で scree 比
$st_k = \frac{(f_k - f_{k-1})/(df_k - df_{k-1})}{(f_{k+1} - f_k)/(df_{k+1} - df_k)}$ を最大にする $k$ を選ぶ。既定の $J$ は
平行分析の結果に 1 を足したもの（`rng` と `n_obs` が要る）で、自由度が正になる範囲に抑える。変数は 6 つ以上。

### `HullResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | 選ばれた因子数 |
| `table` | DataFrame | $k$ ごとの `n_factors`、`fit`、`df`、`st`、`on_hull`、`heywood`、`converged` |
| `fit_index`, `method`, `max_factors` | | 使った設定 |

### `comparison_data`

```text
comparison_data(data, *, rng, max_factors=None, n_population=10000, n_samples=500,
                alpha=0.30, max_iter=50) -> ComparisonDataResult
```

比較データ法（Ruscio & Roche, 2012）。EFAtools の `CD()` と同じ。$k = 1, 2, \dots$ について、観測された Pearson 相関と
各変数の分布を再現する $k$ 個の共通因子をもつ母集団を生成し（Ruscio & Kaczetow, 2008）、データと同じ大きさの標本を
引いて、その固有値と観測固有値の RMSE を記録する。片側の Mann–Whitney 検定が `alpha` で RMSE を下げると言う限り
因子を加える。欠測のない数値データ。

### `ComparisonDataResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `n_factors` | int | 提案される因子数 |
| `rmse` | DataFrame | 標本（行）と試した $k$（列）ごとの固有値の RMSE |
| `p_values` | Series | $k$ と $k - 1$ を比べる順位和の $p$（$k \ge 2$） |
| `population_rmse` | Series | 各母集団が観測相関をどれだけ再現するか |
| `max_factors` | int | 試せた最大の $k$ |

### `detect`

```text
detect(data, clusters) -> DetectResult
```

与えた項目の分割に対する確認的 DETECT（Zhang & Stout, 1999）。各ペアの条件付き共分散は、総得点で条件付けた共分散と
残りの得点で条件付けた共分散の平均。DETECT $= 100 \times$ $\delta_{ij}\,\mathrm{ccov}_{ij}$ の平均（同じクラスタなら
$\delta = 1$、異なれば $-1$）で、ASSI と RATIO（Zhang, 2007）も求める。`data` は欠測のない数値の項目得点。
`clusters` は項目 → ラベルの対応、または列の順のラベル。よく使われる読み方（Jang & Roussos, 2007; Zhang, 2007）は、
DETECT $< 0.2$ で本質的に一次元、$0.2$〜$0.4$ で弱い、$0.4$〜$1$ で中程度、$> 1$ で強い多次元性。ASSI $> 0.25$ または
RATIO $> 0.36$ でその分割について一次元性からの本質的なずれ。

### `DetectResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `indices` | DataFrame | 行 `DETECT`, `ASSI`, `RATIO`, `MADCOV100`, `MCOV100`、列 `unweighted`, `weighted` |
| `detect`, `assi`, `ratio` | float | 重みなしの指標 |
| `conditional_covariance` | DataFrame | 項目 x 項目の条件付き共分散 |
| `clusters` | Series | 使った分割 |
| `n_obs` | int | 回答者数 |

### `exploratory_detect`

```text
exploratory_detect(data, *, max_clusters=4, rng=None,
                   estimation_rows=None) -> ExploratoryDetectResult
```

行を推定用と検証用の半分に分け（`rng` で無作為に、または `estimation_rows` で指定）、推定用の半分で
$\max(\mathrm{ccov}) - \mathrm{ccov}$ に Ward 法を当てて項目をクラスタに分け、$2, \dots,$ `max_clusters` 個への各分割の
DETECT・ASSI・RATIO を両方の半分で報告する。

### `ExploratoryDetectResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `table` | DataFrame | クラスタ数ごとの `cluster_sizes` と、推定用・検証用の半分での `DETECT` / `ASSI` / `RATIO` |
| `clusters` | dict[int, Series] | クラスタ数ごとの分割 |
| `n_clusters` | int | 推定用の半分での DETECT が最大のクラスタ数 |
| `estimation_rows`, `validation_rows` | ndarray | 2 つの半分の行位置 |

### `dimtest`

```text
dimtest(data, *, rng, assessment_items=None, training_fraction=1/3, n_bootstrap=100,
        min_group=20, max_assessment_size=None) -> DimtestResult
```

欠測のない 2 値データに対する DIMTEST（Stout, 1987; Nandakumar & Stout, 1993; Stout, Froelich & Gao, 2001）。項目を
検定用の部分テスト（AT）と分割用の部分テスト（PT）に分ける。本質的に一次元なら、PT の得点が同じ回答者の中での AT の
項目の共分散は 0 に近い。その標準化した和 $T_L$ を、カーネル平滑化した項目反応関数から生成した 1 次元データでの平均で
補正し、$T = (T_L - \bar T_G)/\sqrt{1 + 1/B}$、$p = 1 - \Phi(T)$ とする。`assessment_items` を与えなければ、回答者の
無作為な `training_fraction` で条件付き共分散（DETECT）により AT を選び、残りで検定する。検定は保守的である
（シミュレーションで第 1 種の誤りは名目水準を下回る）。小さい $p$ は第 2 の次元の強い証拠だが、大きい $p$ は一次元性の
弱い証拠にとどまる。

### `DimtestResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `statistic`, `p_value` | float | $T$ と $1 - \Phi(T)$ |
| `t_l`, `t_g` | float, ndarray | $T_L$ とブートストラップの $T_L$ |
| `assessment_items`, `partitioning_items` | list[str] | AT と PT |
| `n_groups` | int | 使った PT の得点の群の数 |
| `training_rows`, `test_rows` | ndarray | AT を選ぶ行と検定する行 |

## 古典的項目分析

インポート: `from flexpyctt import item_analysis, ...`。モデルを使わない項目統計量、信頼性係数、
選択肢の分析（[ガイド 5 節](guide.ja.md#5-古典的項目分析)）。欠測は NaN。

### `score_responses`

```text
score_responses(data, *, key, missing="incorrect") -> DataFrame
```

正答キーで生の回答を採点する。その項目の正答選択肢を選んでいれば 1、そうでなければ 0
（psych の `score.multiple.choice(score = FALSE)`）。

| 引数 | 型 | 説明 |
|---|---|---|
| `data` | DataFrame / 2 次元配列 | 1 行が 1 回答者、1 列が 1 項目。選んだ選択肢（数値でも文字列でもよい）。NaN / None は欠測 |
| `key` | mapping / Series / sequence | 項目ごとの正答選択肢（項目名で、または列の順に）。`==` で比べるので、データと同じコードと型で書く |
| `missing` | `"incorrect"` / `"keep"` | 無回答を 0 点にする（結果は整数）か NaN のまま残す（結果は浮動小数点）（既定 `"incorrect"`） |

**例外**: `SpecificationError`（キーの項目がデータの項目と一致しない、キーに欠測がある）。**警告**:
`FlexpycttWarning`（誰も選んでいない正答選択肢がある。キーのコードが違うことが多い）。

### `item_analysis`

```text
item_analysis(data, *, correlation="pearson", missing="pairwise") -> DataFrame
```

項目ごとに 1 行: `item`、`n`（観測された得点の数）、`n_categories`（観測された得点の種類数）、`mean`、`sd`
（分母 $n-1$）、`difficulty`（平均を観測された最大得点で割ったもの。最上位のカテゴリに誰も届かない項目では、
小さい最大値で割ることになる。0 / 1 項目では正答率）、`item_total`、
`item_rest`（合計得点との相関と、その項目を除いた合計との相関）、`alpha_if_deleted`（2 項目では NaN）。

| 引数 | 型 | 説明 |
|---|---|---|
| `data` | DataFrame / 2 次元配列 | 項目得点（0 / 1 または順序カテゴリの得点。0 以上）。2 項目以上 |
| `correlation` | `"pearson"` / `"polyserial"` | Pearson 相関（0 / 1 項目では点双列相関）か多分系列相関（0 / 1 項目では双列相関）。多分系列相関は 2 段階推定量 $r\,\sigma_j/\sum_k\phi(\tau_{jk})$（Olsson, Drasgow & Dorans, 1982）で、polycor の `polyserial(ML = FALSE)` と同じ（既定 `"pearson"`） |
| `missing` | `"pairwise"` / `"listwise"` | ペアワイズの共分散と各項目の観測された得点を使うか、欠測のない行だけを使うか（既定 `"pairwise"`） |

相関は項目の共分散行列 $C$ から $c_j/\sqrt{C_{jj}\,\mathbf 1^\top C\mathbf 1}$ と、項目 $j$ を除いた対応する式で
計算する。欠測がなければ合計得点・残りの合計との Pearson 相関に等しい。欠測があるとき、psych の `alpha()` は
`r.drop` は同じ値を返すが、`raw.r` は回答者ごとの観測された得点の和との相関になる。

**例外**: `SpecificationError`（数値でない・負の得点、一定の項目、両方に答えた回答者が 2 人未満の項目の組）。
**警告**: `HeywoodWarning`（多分系列相関の推定値が $[-1, 1]$ の外に出た。$\pm1$ にする）。

### `coefficient_alpha`

```text
coefficient_alpha(data, *, missing="pairwise", confidence_level=0.95) -> AlphaResult
```

α 係数（Cronbach, 1951。0 / 1 項目では KR-20）
$\alpha=\frac{k}{k-1}\bigl(1-\operatorname{tr}C/\mathbf 1^\top C\mathbf 1\bigr)$、相関行列から求める標準化 α、
Feldt の信頼区間 $\bigl[1-(1-\alpha)F_{1-\gamma/2},\,1-(1-\alpha)F_{\gamma/2}\bigr]$（$F$ は自由度
$(n-1,(n-1)(k-1))$ の分位点。Feldt, Woodruff & Salih, 1987）。psych の `alpha()` と一致する。`missing` は
`item_analysis()` と同じ。`"pairwise"` では Feldt の $n$ は 1 つ以上回答した回答者の数。

### `AlphaResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `alpha`, `standardized_alpha` | float | α と標準化 α |
| `confidence_interval`, `confidence_level` | tuple, float | Feldt の（下限, 上限）とその信頼水準 |
| `n_items`, `n_obs` | int | 使った項目数と回答者数（回答が 1 つもない行は除く。psych はすべての行を数える） |
| `if_deleted` | DataFrame | 項目ごとに、その項目を除いた `alpha` と `standardized_alpha`（2 項目では NaN） |

### `coefficient_omega`

```text
coefficient_omega(data, *, correlation="pearson") -> OmegaResult
```

1 因子モデルの ω total。項目の相関行列 $R$ に ULS（minres）で当てはめた負荷量 $\lambda_j$ から
$s_j=\operatorname{sign}(\lambda_j)$ として $\omega_t=1-\sum_j(1-\lambda_j^2)/\mathbf s^\top R\mathbf s$
（McDonald, 1999; Revelle & Zinbarg, 2009）を求める。既定の `flip = TRUE` での psych の
`omega(nfactors = 1)$omega.tot` と同じ。つまり負の負荷量の項目は反転する（`FlexpycttWarning` を出し、
`reversed_items` に挙げる）。採点したままの合計の ω が欲しいときは、先にその項目を逆転採点するか除く。`correlation` は `"pearson"`（ペアワイズ）か
`"polychoric"`（既定の設定の [`polychoric`](#polychoric)）。正定値でない行列は平滑化し、
`NotPositiveDefiniteWarning` を出す。3 項目以上。階層 ω は提供しない。

### `OmegaResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `omega` | float | $\omega_t$ |
| `loadings`, `uniquenesses` | Series | 1 因子の負荷量（和が非負になる向き）と $1-\lambda_j^2$ |
| `correlation` | DataFrame | 因子を当てはめた相関行列（採点したままの項目） |
| `reversed_items` | tuple of str | 負の負荷量を持ち、`omega` の計算前に反転した項目 |

### `distractor_analysis`

```text
distractor_analysis(data, *, key, n_groups=3) -> DataFrame
```

多肢選択の回答の選択肢分析。`key` で採点し（無回答は 0 点）、合計得点 $T$ を求める。項目と選択肢（観測された
選択肢と正答）の組ごとに 1 行: `item`、`option`、`is_key`、`n`、`proportion`（その項目に答えた人のうちの割合）、
`point_biserial`（その選択肢を選んだかどうかと残りの合計 $T - x_j$ との点双列相関）、`mean_score`（選んだ人の
$T$ の平均）、`proportion_group_1` … `proportion_group_<n_groups>`（得点の低い群から高い群へ、各群で選んだ割合）。
群は $T$ の分位点で、R の `cut(T, quantile(T, 0:G / G), include.lowest = TRUE)` と同じく区切る（同点は同じ群に
入る）。分位点が重なると `SpecificationError`（群の数を減らす）。

## 観測得点による特異項目機能（DIF）

インポート: `from flexpyctt import mantel_haenszel, ...`。IRT モデルを使わない DIF の検出で、回答者を
観測得点でそろえ（マッチング）、マッチング得点の値ごとに群を比べる。推定済みの多母集団モデルによる DIF の検定は
flexpyirt の [`FitResult.dif`](https://takuizum.github.io/flexpyirt/ja/reference/#fitresult) を参照。

4 つの関数に共通の引数:

| 引数 | 既定値 | 説明 |
|---|---|---|
| `data` | | 回答者 x 項目（DataFrame または配列）。検定する項目とマッチングに使う項目に欠測があってはならない（先にその行を除く） |
| `group` | | 各回答者の群のラベル |
| `focal` | | 焦点群のラベル（`generalized_mantel_haenszel` ではラベルのリスト）。それ以外の回答者はすべて参照群 |
| `items` | `None` | 検定する項目（既定ではすべて） |
| `anchor_items` | `None` | 合計をマッチング得点にする項目（既定ではすべて）。検定する項目は常にそれ自身のマッチング得点に含まれる |
| `matching` | `None` | 合計得点の代わりに使う外部のマッチング変数（回答者ごとに 1 つ）。`anchor_items`・`purify` とは併用できない |
| `purify` | `False` | 純化: ある段階で検出された項目を次の段階のマッチング得点から外し、検出される項目の組が繰り返すか `max_iter` に達するまで続ける |
| `max_iter` | `10` | 純化の反復の上限 |
| `p_adjust` | `"none"` | `"holm"` または `"BH"`: 最終的な p 値を検定した項目にわたって補正する。純化には補正前の p 値を使う |
| `alpha` | `0.05` | `flagged`、純化、ETS 分類の有意水準 |

マッチング得点の異なる値はそれぞれ 1 つの層になる（得点をまとめることはしない）。

### `mantel_haenszel`

```text
mantel_haenszel(data, *, group, focal, items=None, anchor_items=None, matching=None,
                purify=False, max_iter=10, p_adjust="none", alpha=0.05,
                continuity_correction=True, classification="ets") -> DIFResult
```

0 / 1 項目・2 群の Mantel–Haenszel 検定（Holland & Thayer, 1988）。マッチング得点 $k$ で正答・誤答した参照群の
人数を $A_k, B_k$、焦点群の人数を $C_k, D_k$、$k$ の人数を $T_k$ として

$$\chi^2_{MH}=\frac{\big(\lvert\sum_kA_k-\sum_kE(A_k)\rvert-\tfrac12\big)^2}{\sum_k\operatorname{Var}(A_k)},\qquad
\hat\alpha_{MH}=\frac{\sum_kA_kD_k/T_k}{\sum_kB_kC_k/T_k},\qquad \Delta_{MH}=-2.35\ln\hat\alpha_{MH}.$$

$\hat\alpha_{MH}>1$（$\Delta_{MH}<0$）は項目が参照群に有利であることを表す。$\ln\hat\alpha_{MH}$ の標準誤差は
Phillips & Holland (1987) による。`continuity_correction=False` で $\tfrac12$ を引かない。ETS 分類
（`classification="ets"`）: $\lvert\Delta_{MH}\rvert<1$ または $\chi^2_{MH}$ が有意でなければ（`adj_p` で判定するので、
B と C の項目は必ず `flagged`）**A**、$\lvert\Delta_{MH}\rvert\ge1.5$ かつ片側検定で 1 より有意に大きければ
（$(\lvert\Delta_{MH}\rvert-1)/SE>z_{1-\alpha}$。`alpha=0.05` で 1.645）**C**、
それ以外は **B**。`classification="effect_size"` は閾値 1 と 1.5 だけを使う（difR が表示する分類）。

`table` の列: `item`, `chi2`, `odds_ratio`, `log_odds_ratio`, `log_odds_ratio_se`, `delta`, `delta_se`,
`ets_class`, `p`, `adj_p`, `flagged`。

### `mantel`

```text
mantel(data, *, group, focal, items=None, anchor_items=None, matching=None, purify=False,
       max_iter=10, p_adjust="none", alpha=0.05) -> DIFResult
```

順序のある多値項目・2 群の Mantel (1963) の検定。焦点群の項目得点の和を、マッチング得点を与えたときの期待値と
比べる（$df=1$。0 / 1 項目では連続修正なしの $\chi^2_{MH}$ に等しい）。効果量は Liu & Agresti (1996) の累積共通
オッズ比で、$\hat\alpha_{MH}$ と同じ向き（1 より大きいと参照群に有利。0 / 1 項目では $\hat\alpha_{MH}$ に等しい）に
し、その対数の Liu & Agresti (1996) の式 (3) による標準誤差をつける。層が少数で大きくても、（合計得点でマッチング
したときのように）小さな層が多数あっても妥当な推定量である。

`table` の列: `item`, `chi2`, `df`, `cumulative_odds_ratio`, `log_cumulative_odds_ratio`,
`log_cumulative_odds_ratio_se`, `p`, `adj_p`, `flagged`。

### `generalized_mantel_haenszel`

```text
generalized_mantel_haenszel(data, *, group, focal, items=None, anchor_items=None,
                            matching=None, purify=False, max_iter=10, p_adjust="none",
                            alpha=0.05) -> DIFResult
```

複数の焦点群（`focal` はリスト）や順序のないカテゴリのための一般化 Mantel–Haenszel 検定（Somes, 1986;
Penfield, 2001）。各層の群 x カテゴリの表を周辺度数を与えたときの期待値と比べ、$G$ 群・観測された $C$ カテゴリ
で $df=(G-1)(C-1)$。焦点群が 1 つで 0 / 1 項目なら連続修正なしの $\chi^2_{MH}$ に等しい。

`table` の列: `item`, `chi2`, `df`, `p`, `adj_p`, `flagged`。

### `logistic_dif`

```text
logistic_dif(data, *, group, focal, items=None, anchor_items=None, matching=None,
             purify=False, max_iter=10, p_adjust="none", alpha=0.05, kind="both",
             test="likelihood_ratio") -> DIFResult
```

0 / 1 項目・2 群のロジスティック回帰による DIF（Swaminathan & Rogers, 1990）。マッチング得点を $S$、焦点群の
指示変数を $G$ として、モデル $\operatorname{logit}P(x=1)=\beta_0+\beta_1S$ $(M_0)$、$+\beta_2G$ $(M_1)$、
$+\beta_3SG$ $(M_2)$ を最尤法で推定する。`kind="both"` は $M_0$ と $M_2$（$df=2$）、`"uniform"` は $M_0$ と
$M_1$、`"non_uniform"` は $M_1$ と $M_2$（$df=1$）を比べる。`test` は `"likelihood_ratio"` または `"wald"`
（大きいほうのモデルで加えた係数の Wald 検定）。効果量は Nagelkerke の $R^2$ の差（Zumbo, 1999）で、
$\Delta R^2$ だけによる Zumbo & Thomas（区切り 0.13、0.26）と Jodoin & Gierl（0.035、0.07）の A / B / C 分類を
つける。外部の `matching` は連続変数でもよい。

`table` の列: `item`, `chi2`, `df`, `delta_r2`, `zumbo_thomas_class`, `jodoin_gierl_class`,
`group_coefficient`, `group_coefficient_se`, `interaction_coefficient`, `interaction_coefficient_se`
（大きいほうのモデルの $G$ と $SG$ の係数。その項がなければ NaN）, `p`, `adj_p`, `flagged`。回帰が収束しない
とき（分離など）は `ConvergenceWarning` を出す。

### `DIFResult`

| 属性 | 型 | 説明 |
|---|---|---|
| `table` | DataFrame | 検定した項目ごとに 1 行（列は上の各関数を参照）。`flagged` は `adj_p < alpha` |
| `flagged_items` | list[str] | `table` で検出された項目 |
| `anchor_items` | list[str] | 最後の段階でマッチング得点を作った項目（`matching` を使ったときは空） |
| `n_iter`, `converged` | int, bool | 純化の反復回数と、検出された項目の組が繰り返したか（しなければ `ConvergenceWarning`） |
| `flag_history` | DataFrame | 各段階で検出された項目（補正前の $p<$ `alpha`）。0 行目が最初の検定 |
| `method` | str | `"mantel_haenszel"`, `"mantel"`, `"generalized_mantel_haenszel"`, `"logistic"` |

---

## 例外と警告

### `flexpyctt.errors`

`from flexpyctt.errors import ...` で使う。

#### `FlexpycttError`

flexpyctt が意図して送出するすべての例外の基底クラス。

#### `SpecificationError`

指定がそれ自体と、またはデータと矛盾している。範囲外のオプション、数値でない反応、カテゴリが 1 つしか観測されていない
項目など。計算を始める前に送出する。`FlexpycttError` と `ValueError` のサブクラス。

#### `FlexpycttWarning`

flexpyctt が出すすべての警告の基底クラス（`UserWarning` のサブクラス）。

#### `ConvergenceWarning`

収束基準を満たさずに推定が止まった（`max_iter` に達した場合など）。

#### `HeywoodWarning`

境界解に達した（相関が 1、独自性が境界にある、など）。

#### `NotPositiveDefiniteWarning`

相関行列（ペアごとに推定したポリコリック行列など）が正定値でなかったので、正定値の行列に平滑化した。
