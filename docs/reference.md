# API reference

English | [日本語](reference.ja.md)

The public API of flexpyctt (everything importable with `from flexpyctt import ...`). For the
workflow see the [user guide](guide.md). The docstrings carry the same information plus the
formulas and sources (`help(flexpyctt.polychoric)` and so on).

Apart from the first one or two, arguments of public functions are keyword-only (after `*`).
Functions that draw random numbers take `rng` (an integer seed or a `numpy.random.Generator`).
[flexpyirt](https://takuizum.github.io/flexpyirt/) re-exports the names listed here under the same names, except `polychoric`:
`flexpyirt.polychoric` is a JAX implementation of the same estimator.

- [`polychoric`](#polychoric), [`PolychoricResult`](#polychoricresult), [`parallel_analysis`](#parallel_analysis), [`ParallelAnalysisResult`](#parallelanalysisresult), [`velicer_map`](#velicer_map), [`MAPResult`](#mapresult), [`empirical_kaiser`](#empirical_kaiser), [`EmpiricalKaiserResult`](#empiricalkaiserresult), [`efa`](#efa), [`EFAResult`](#efaresult), [`hull`](#hull), [`HullResult`](#hullresult), [`comparison_data`](#comparison_data), [`ComparisonDataResult`](#comparisondataresult), [`detect`](#detect), [`DetectResult`](#detectresult), [`exploratory_detect`](#exploratory_detect), [`ExploratoryDetectResult`](#exploratorydetectresult), [`dimtest`](#dimtest), [`DimtestResult`](#dimtestresult), [`score_responses`](#score_responses), [`item_analysis`](#item_analysis), [`coefficient_alpha`](#coefficient_alpha), [`AlphaResult`](#alpharesult), [`coefficient_omega`](#coefficient_omega), [`OmegaResult`](#omegaresult), [`distractor_analysis`](#distractor_analysis), [`mantel_haenszel`](#mantel_haenszel), [`mantel`](#mantel), [`generalized_mantel_haenszel`](#generalized_mantel_haenszel), [`logistic_dif`](#logistic_dif), [`DIFResult`](#difresult), [`flexpyctt.errors`](#flexpyctterrors)

---

## Correlations

Import: `from flexpyctt import polychoric`.

### `polychoric`

```text
polychoric(data, *, smooth=True, zero_cell_correction=0.5, tol=1e-10,
           max_iter=100) -> PolychoricResult
```

Polychoric correlation matrix (tetrachoric for binary items) of ordinal responses, by two-step
maximum likelihood ([guide section 2](guide.md#2-polychoric-correlations)): each item's
thresholds come from its marginal proportions, then each pair's correlation maximizes the
likelihood of the pair's table under a discretized bivariate normal (Olsson, 1979). The
bivariate normal probabilities use Owen's (1956) $T$ function. Agrees with lavaan's `lavCor` on
ordered variables.

`flexpyirt.polychoric` has the same arguments and result and computes the same estimator with JAX
(bivariate normal probabilities by Genz's, 2004, algorithm); it also raises `PrecisionError` when
JAX's 64-bit mode is disabled. The two agree to about $10^{-9}$.

| Argument | Type | Description |
|---|---|---|
| `data` | DataFrame / 2-D array | One row per respondent, one column per item. Ordered category codes (any numbers; ascending order is the category order); missing as NaN. Each item needs at least two observed categories; unobserved codes are not categories |
| `smooth` | bool | Smooth a matrix that is not positive definite (psych's `cor.smooth` rule) with a `NotPositiveDefiniteWarning` (default `True`) |
| `zero_cell_correction` | float | For a 2 x 2 table with exactly one empty cell, add this to the diagonal containing the empty cell and subtract it from the other diagonal, keeping the margins (lavaan's default rule). Without it the estimate is $\pm 1$. `0` disables it (default `0.5`) |
| `tol` | float | Convergence criterion on the gradient per observation (default `1e-10`) |
| `max_iter` | int | Maximum Newton iterations per pair (default `100`) |

Missing responses use available cases: thresholds use every observed response to the item, and a
pair's table uses respondents who answered both items.

**Returns**: [`PolychoricResult`](#polychoricresult)

**Raises**: `SpecificationError` (non-numeric data, fewer
than two items, an item with one observed category, a pair with no respondent answering both, or
one item constant among those who did).

**Warns**: `HeywoodWarning` (a pair's likelihood is largest at $|\rho| \to 1$),
`ConvergenceWarning` (a pair did not converge), `NotPositiveDefiniteWarning` (the matrix was
smoothed).

### `PolychoricResult`

Returned by `polychoric()`. Immutable.

| Attribute | Type | Description |
|---|---|---|
| `correlation` | DataFrame | Item x item correlation matrix (smoothed when `smooth=True` and needed) |
| `unsmoothed_correlation` | DataFrame | The pairwise estimates before smoothing |
| `thresholds` | dict[str, ndarray] | Per item, the $K - 1$ thresholds $\tau_k = \Phi^{-1}(P(Y \le k))$ of its $K$ observed categories |
| `categories` | dict[str, ndarray] | Per item, the observed response values in ascending order |
| `n_obs` | DataFrame | Respondents with both responses observed (diagonal: responses to the item) |
| `converged`, `at_boundary`, `zero_cell_corrected` | DataFrame (bool) | Per pair: convergence, maximum at $\lvert\rho\rvert \to 1$ (the estimate is then $\pm\tanh 8$), whether `zero_cell_correction` was applied |
| `smoothed` | bool | Whether smoothing changed the matrix |
| `min_eigenvalue` | float | Smallest eigenvalue of `unsmoothed_correlation` |

## Number of dimensions

These answer "how many dimensions?" as diagnostics; combine several rather than trusting one
([guide section 3](guide.md#3-number-of-dimensions)).

The tools that need a fitted model (`model_parallel_analysis`, `dimensionality`, the local
dependence statistics) are in [flexpyirt](https://takuizum.github.io/flexpyirt/reference/).

### `parallel_analysis`

```text
parallel_analysis(data, *, rng, correlation="polychoric", eigen_type="pca",
                  reference="permutation", n_replications=100, quantile=0.95,
                  zero_cell_correction=0.5) -> ParallelAnalysisResult
```

Horn's (1965) parallel analysis: the leading eigenvalues of the observed correlation matrix that
exceed the `quantile` (Glorfeld, 1995) of eigenvalues from reference data without common factors
are counted.

| Argument | Type | Description |
|---|---|---|
| `data` | DataFrame / 2-D array | Respondents x items (ordered codes, or continuous scores for `correlation="pearson"`); NaN is missing |
| `rng` | int / `numpy.random.Generator` | Seed or generator for the reference data, e.g. `rng=0` |
| `correlation` | `"polychoric"` / `"pearson"` | Correlation of the data and of permutation replications (both smoothed when not positive definite) |
| `eigen_type` | `"pca"` / `"smc"` | Eigenvalues of the correlation matrix, or of the reduced matrix with squared multiple correlations on the diagonal |
| `reference` | `"permutation"` / `"normal"` | Permute each item independently (keeps the marginals; Garrido, Abad & Ponsoda, 2013), or i.i.d. normal data of the same size (Pearson) |
| `n_replications` | int | Number of reference data sets (default `100`) |
| `quantile` | float / None | Quantile in (0, 1) of the reference eigenvalues (default `0.95`); `None` uses the mean |
| `zero_cell_correction` | float | Passed to `polychoric()` |

**Returns**: [`ParallelAnalysisResult`](#parallelanalysisresult)

**Raises**: `SpecificationError` (invalid options or data, fewer than 3 items).

**Warns**: `FlexpyirtWarning` when permutation replications had to be skipped (missing data
leaving a permuted pair without usable responses).

### `ParallelAnalysisResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | Number of leading observed eigenvalues above `reference` |
| `observed` | ndarray | Observed eigenvalues, descending |
| `reference`, `reference_mean` | ndarray | Reference quantile (or mean) and mean per position |
| `replications` | ndarray | `(n_replications - n_skipped, p)` reference eigenvalues |
| `correlation` | DataFrame | The observed correlation matrix |
| `eigen_type`, `reference_kind`, `correlation_kind`, `quantile`, `n_skipped` | | Settings used and skipped replications |

### `velicer_map`

```text
velicer_map(correlation, *, power=4) -> MAPResult
```

Velicer's minimum average partial test: after partialling out the first $m$ principal
components, the average of the off-diagonal partial correlations raised to `power` is minimized
over $m = 0, \dots, p - 1$. `power=2` is the original test (Velicer, 1976), `power=4` the revised
one (Velicer, Eaton & Fava, 2000). `correlation` is a matrix (DataFrame / array) or a
`PolychoricResult`.

### `MAPResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | The minimizing $m$ |
| `statistic` | ndarray | Average partial correlation statistic for $m = 0, \dots, p - 1$ (NaN once a partial variance is 0) |
| `power` | int | 2 or 4 |

### `empirical_kaiser`

```text
empirical_kaiser(correlation, *, n_obs=None) -> EmpiricalKaiserResult
```

Empirical Kaiser criterion (Braeken & van Assen, 2017): the $j$-th eigenvalue is compared with
$\max\{(1+\sqrt{p/n})^2 (p-\sum_{i<j}\lambda_i)/(p-j+1),\ 1\}$. `n_obs` is required unless
`correlation` is a `PolychoricResult` (whose smallest pairwise $n$ is used).

### `EmpiricalKaiserResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | Number of leading eigenvalues above the reference |
| `eigenvalues`, `reference` | ndarray | Eigenvalues and reference values |
| `n_obs` | int | Sample size used |

### `efa`

```text
efa(correlation, *, n_factors, method="uls", n_obs=None) -> EFAResult
```

Unrotated exploratory factor analysis of a correlation matrix (DataFrame, array or
`PolychoricResult`) by ULS (minres; Harman & Jones, 1966) or ML (Jöreskog, 1967), minimizing over
the uniquenesses from $1 - \mathrm{SMC}$ within $[0.005, \max(1, \mathrm{SMC})]$, as psych's
`fa(rotate = "none")` and EFAtools' `EFA(rotation = "none")`. With `n_obs` (default for a
`PolychoricResult`: its smallest pairwise $n$) the fit includes the Bartlett-corrected $\chi^2$, its
$p$-value, CFI and RMSEA; CAF and $df$ are always reported. Warns `HeywoodWarning` when a
uniqueness reaches its bound and `ConvergenceWarning` when the optimizer stops early.

### `EFAResult`

| Attribute | Type | Description |
|---|---|---|
| `loadings` | DataFrame | Unrotated loadings, columns signed to non-negative sums |
| `uniquenesses`, `communalities` | Series | $\psi_i$ and $h_i^2$ |
| `fit` | dict | `caf`, `df` and, with `n_obs`, `chi2`, `p`, `cfi`, `rmsea` |
| `method`, `converged`, `heywood`, `objective` | | Estimator and diagnostics |

### `hull`

```text
hull(correlation, *, n_obs=None, max_factors=None, rng=None, method="uls",
     fit_index="caf", n_replications=1000) -> HullResult
```

Hull method (Lorenzo-Seva, Timmerman & Kiers, 2011), as EFAtools' `HULL()`: solutions with
$k = 0, \dots, J$ factors are placed by fit (CAF, CFI or $1-$RMSEA) and degrees of freedom; on the
upper convex boundary the selected $k$ maximizes the scree ratio
$st_k = \frac{(f_k - f_{k-1})/(df_k - df_{k-1})}{(f_{k+1} - f_k)/(df_{k+1} - df_k)}$. By default
$J$ is one more than parallel analysis (needs `rng` and `n_obs`), capped for positive degrees of
freedom. At least 6 variables.

### `HullResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | Selected number of factors |
| `table` | DataFrame | `n_factors`, `fit`, `df`, `st`, `on_hull`, `heywood`, `converged` per $k$ |
| `fit_index`, `method`, `max_factors` | | Settings used |

### `comparison_data`

```text
comparison_data(data, *, rng, max_factors=None, n_population=10000, n_samples=500,
                alpha=0.30, max_iter=50) -> ComparisonDataResult
```

Comparison-data method (Ruscio & Roche, 2012), as EFAtools' `CD()`. For $k = 1, 2, \dots$ a
population with $k$ common factors that reproduces the observed Pearson correlations and every
variable's distribution is generated (Ruscio & Kaczetow, 2008); samples of the data's size are
drawn and the RMSE between their eigenvalues and the observed ones is recorded. A factor is added
while a one-sided Mann–Whitney test at `alpha` says it lowers the RMSE. Complete numeric data.

### `ComparisonDataResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | Suggested number of factors |
| `rmse` | DataFrame | Eigenvalue RMSE per sample (rows) and tested $k$ (columns) |
| `p_values` | Series | Rank-sum $p$ for $k$ vs $k - 1$ ($k \ge 2$) |
| `population_rmse` | Series | How well each population reproduces the observed correlations |
| `max_factors` | int | Largest $k$ that could be tested |

### `detect`

```text
detect(data, clusters) -> DetectResult
```

Confirmatory DETECT (Zhang & Stout, 1999) for a given partition of the items. The conditional
covariance of each pair is the average of its covariances given the total score and given the
rest score; DETECT $= 100 \times$ the mean of $\delta_{ij}\,\mathrm{ccov}_{ij}$ ($\delta = 1$
within a cluster, $-1$ across), with ASSI and RATIO (Zhang, 2007). `data` must be complete
numeric item scores; `clusters` is a mapping item → label or labels in column order. A common
reading (Jang & Roussos, 2007; Zhang, 2007): DETECT $< 0.2$ essentially unidimensional,
$0.2$–$0.4$ weak, $0.4$–$1$ moderate, $> 1$ strong; ASSI $> 0.25$ or RATIO $> 0.36$ an essential
deviation for the partition.

### `DetectResult`

| Attribute | Type | Description |
|---|---|---|
| `indices` | DataFrame | Rows `DETECT`, `ASSI`, `RATIO`, `MADCOV100`, `MCOV100`; columns `unweighted`, `weighted` |
| `detect`, `assi`, `ratio` | float | Unweighted indices |
| `conditional_covariance` | DataFrame | Item x item conditional covariances |
| `clusters` | Series | Partition used |
| `n_obs` | int | Respondents |

### `exploratory_detect`

```text
exploratory_detect(data, *, max_clusters=4, rng=None,
                   estimation_rows=None) -> ExploratoryDetectResult
```

Splits the rows into an estimation half and a validation half (`rng` for a random split, or
`estimation_rows`), clusters the items on the estimation half by Ward's method on
$\max(\mathrm{ccov}) - \mathrm{ccov}$, and reports DETECT, ASSI and RATIO of every partition
into $2, \dots,$ `max_clusters` clusters on both halves.

### `ExploratoryDetectResult`

| Attribute | Type | Description |
|---|---|---|
| `table` | DataFrame | Per number of clusters: `cluster_sizes` and `DETECT` / `ASSI` / `RATIO` on the estimation and validation halves |
| `clusters` | dict[int, Series] | Partition for each number of clusters |
| `n_clusters` | int | Number of clusters with the largest estimation-half DETECT |
| `estimation_rows`, `validation_rows` | ndarray | Row positions of the halves |

### `dimtest`

```text
dimtest(data, *, rng, assessment_items=None, training_fraction=1/3, n_bootstrap=100,
        min_group=20, max_assessment_size=None) -> DimtestResult
```

DIMTEST (Stout, 1987; Nandakumar & Stout, 1993; Stout, Froelich & Gao, 2001) for complete binary
data. The items are split into an assessment subtest (AT) and a partitioning subtest (PT); given
the PT score, the AT items' covariances are near zero under essential unidimensionality. Their
standardized sum $T_L$ is corrected by its mean in unidimensional data generated from
kernel-smoothed item response functions, $T = (T_L - \bar T_G)/\sqrt{1 + 1/B}$, and
$p = 1 - \Phi(T)$. Without `assessment_items` the AT is chosen by conditional covariances
(DETECT) on a random `training_fraction` of the respondents and the test uses the rest. The test
is conservative (type I error below the nominal level in simulations): a small $p$ is strong
evidence of a second dimension, a large $p$ only weak evidence for unidimensionality.

### `DimtestResult`

| Attribute | Type | Description |
|---|---|---|
| `statistic`, `p_value` | float | $T$ and $1 - \Phi(T)$ |
| `t_l`, `t_g` | float, ndarray | $T_L$ and the bootstrap $T_L$ values |
| `assessment_items`, `partitioning_items` | list[str] | AT and PT |
| `n_groups` | int | PT-score groups used |
| `training_rows`, `test_rows` | ndarray | Rows used to choose the AT and to test |

## Classical item analysis

Import: `from flexpyctt import item_analysis, ...`. Model-free item statistics,
reliability coefficients and option analysis ([guide section 5](guide.md#5-classical-item-analysis)).
Missing responses are NaN.

### `score_responses`

```text
score_responses(data, *, key, missing="incorrect") -> DataFrame
```

Score raw responses with an answer key: 1 if the respondent chose the item's keyed option, else 0
(psych's `score.multiple.choice(score = FALSE)`).

| Argument | Type | Description |
|---|---|---|
| `data` | DataFrame / 2-D array | One row per respondent, one column per item; chosen options in any codes (numbers or strings); NaN / None is missing |
| `key` | mapping / Series / sequence | Correct option per item (by item name, or in column order). Compared with `==`, so the key must use the data's codes and types |
| `missing` | `"incorrect"` / `"keep"` | Score a missing response 0 (integer result) or keep it NaN (float result) (default `"incorrect"`) |

**Raises**: `SpecificationError` (the key does not name exactly the data's items, or a key entry
is missing). **Warns**: `FlexpycttWarning` (nobody chose some item's keyed option, often a key in
the wrong codes).

### `item_analysis`

```text
item_analysis(data, *, correlation="pearson", missing="pairwise") -> DataFrame
```

One row per item: `item`, `n` (observed scores), `n_categories` (distinct observed scores),
`mean`, `sd` ($n-1$ denominator), `difficulty` (mean divided by the largest observed score, so
an item whose top category nobody reached is divided by a smaller maximum; the proportion correct
for 0 / 1 items), `item_total`, `item_rest` (correlation with the total score
and with the total of the other items) and `alpha_if_deleted` (NaN with two items).

| Argument | Type | Description |
|---|---|---|
| `data` | DataFrame / 2-D array | Item scores (0 / 1 or ordered category scores, at least 0); at least two items |
| `correlation` | `"pearson"` / `"polyserial"` | Pearson (point-biserial for 0 / 1 items) or polyserial (biserial for 0 / 1 items) correlations; the polyserial is the two-step estimator $r\,\sigma_j/\sum_k\phi(\tau_{jk})$ (Olsson, Drasgow & Dorans, 1982), polycor's `polyserial(ML = FALSE)` (default `"pearson"`) |
| `missing` | `"pairwise"` / `"listwise"` | Pairwise-complete covariances and each item's observed scores, or complete rows only (default `"pairwise"`) |

The correlations are computed from the item covariance matrix $C$:
$c_j/\sqrt{C_{jj}\,\mathbf 1^\top C\mathbf 1}$ and its counterpart without item $j$, which equal the
Pearson correlations with the total and rest scores on complete data. With missing data psych's
`alpha()` reports the same `r.drop`, but its `raw.r` correlates with each respondent's sum of
observed scores.

**Raises**: `SpecificationError` (non-numeric or negative scores, a constant item, a pair of items
with fewer than two joint observations). **Warns**: `HeywoodWarning` (a polyserial estimate
outside $[-1, 1]$, set to $\pm1$).

### `coefficient_alpha`

```text
coefficient_alpha(data, *, missing="pairwise", confidence_level=0.95) -> AlphaResult
```

Coefficient alpha (Cronbach, 1951; KR-20 for 0 / 1 items),
$\alpha=\frac{k}{k-1}\bigl(1-\operatorname{tr}C/\mathbf 1^\top C\mathbf 1\bigr)$, standardized alpha
from the correlation matrix, and Feldt's interval
$\bigl[1-(1-\alpha)F_{1-\gamma/2},\,1-(1-\alpha)F_{\gamma/2}\bigr]$ with $F$ quantiles on
$(n-1,(n-1)(k-1))$ degrees of freedom (Feldt, Woodruff & Salih, 1987). Agrees with psych's
`alpha()`. `missing` as in `item_analysis()`; under `"pairwise"` the Feldt $n$ is the number of
respondents with at least one response.

### `AlphaResult`

| Attribute | Type | Description |
|---|---|---|
| `alpha`, `standardized_alpha` | float | Alpha and standardized alpha |
| `confidence_interval`, `confidence_level` | tuple, float | Feldt's (lower, upper) interval and its level |
| `n_items`, `n_obs` | int | Items and respondents used (rows without any response are dropped; psych counts every row) |
| `if_deleted` | DataFrame | Per item, `alpha` and `standardized_alpha` without it (NaN with two items) |

### `coefficient_omega`

```text
coefficient_omega(data, *, correlation="pearson") -> OmegaResult
```

Omega total of a one-factor model: ULS (minres) loadings $\lambda_j$ fitted to the item
correlation matrix $R$, and $\omega_t=1-\sum_j(1-\lambda_j^2)/\mathbf s^\top R\mathbf s$ with
$s_j=\operatorname{sign}(\lambda_j)$ (McDonald, 1999; Revelle & Zinbarg, 2009), psych's
`omega(nfactors = 1)$omega.tot` with its default `flip = TRUE`. Items that load negatively are
thus reversed (with a `FlexpycttWarning` and listed in `reversed_items`); reverse-score or drop
them first to get the omega of the total as scored. `correlation` is
`"pearson"` (pairwise-complete) or `"polychoric"` ([`polychoric`](#polychoric) with its defaults);
a matrix that is not positive definite is smoothed with a `NotPositiveDefiniteWarning`. At least
three items. Hierarchical omega is not provided.

### `OmegaResult`

| Attribute | Type | Description |
|---|---|---|
| `omega` | float | $\omega_t$ |
| `loadings`, `uniquenesses` | Series | One-factor loadings (signed to a non-negative sum) and $1-\lambda_j^2$ |
| `correlation` | DataFrame | The correlation matrix the factor was fitted to (items as scored) |
| `reversed_items` | tuple of str | Items with a negative loading, reversed before computing `omega` |

### `distractor_analysis`

```text
distractor_analysis(data, *, key, n_groups=3) -> DataFrame
```

Option analysis of multiple-choice responses. Responses are scored with `key` (missing scores 0)
to give the total score $T$. One row per item and option (observed options and the key):
`item`, `option`, `is_key`, `n`, `proportion` (of the item's respondents), `point_biserial`
(choosing the option against the rest score $T - x_j$), `mean_score` (mean $T$ of those who chose
it) and `proportion_group_1` ... `proportion_group_<n_groups>`: the proportion choosing it in
total-score groups from low to high, cut at the quantiles of $T$ as R's
`cut(T, quantile(T, 0:G / G), include.lowest = TRUE)` (ties stay in one group). Raises
`SpecificationError` when the quantiles are not distinct (use fewer groups).

## Differential item functioning (observed scores)

Import: `from flexpyctt import mantel_haenszel, ...`. DIF screening without an IRT
model: respondents are matched on an observed score and the groups are compared within each
matching-score level. For DIF tests based on a fitted multiple-group model, see
flexpyirt's [`FitResult.dif`](https://takuizum.github.io/flexpyirt/reference/#fitresult).

Arguments shared by the four functions:

| Argument | Default | Description |
|---|---|---|
| `data` | | Respondents x items (DataFrame or array). No missing values on the studied and matching items (drop those rows first) |
| `group` | | Group label of each respondent |
| `focal` | | Label of the focal group (`generalized_mantel_haenszel`: a list of labels). Every other respondent is in the reference group |
| `items` | `None` | Items to test (default: all) |
| `anchor_items` | `None` | Items whose total is the matching score (default: all). The studied item is always part of its own matching score |
| `matching` | `None` | An external matching variable (one value per respondent) instead of the total score; not with `anchor_items` or `purify` |
| `purify` | `False` | Purification: items flagged at one step leave the matching score of the next, until the flagged set repeats or `max_iter` is reached |
| `max_iter` | `10` | Maximum number of purification iterations |
| `p_adjust` | `"none"` | `"holm"` or `"BH"`: adjustment of the final p-values over the tested items. Purification uses unadjusted p-values |
| `alpha` | `0.05` | Significance level for `flagged`, purification and the ETS classes |

Each distinct matching-score value is a stratum (no thick matching).

### `mantel_haenszel`

```text
mantel_haenszel(data, *, group, focal, items=None, anchor_items=None, matching=None,
                purify=False, max_iter=10, p_adjust="none", alpha=0.05,
                continuity_correction=True, classification="ets") -> DIFResult
```

Mantel–Haenszel test for 0 / 1 items and two groups (Holland & Thayer, 1988). With $A_k, B_k$
the reference respondents answering correctly / incorrectly at matching score $k$, $C_k, D_k$
the focal ones and $T_k$ the number of respondents at $k$:

$$\chi^2_{MH}=\frac{\big(\lvert\sum_kA_k-\sum_kE(A_k)\rvert-\tfrac12\big)^2}{\sum_k\operatorname{Var}(A_k)},\qquad
\hat\alpha_{MH}=\frac{\sum_kA_kD_k/T_k}{\sum_kB_kC_k/T_k},\qquad \Delta_{MH}=-2.35\ln\hat\alpha_{MH}.$$

$\hat\alpha_{MH}>1$ ($\Delta_{MH}<0$) means the item favours the reference group. The SE of
$\ln\hat\alpha_{MH}$ is Phillips & Holland's (1987). `continuity_correction=False` drops the
$\tfrac12$. ETS classes (`classification="ets"`): **A** if $\lvert\Delta_{MH}\rvert<1$ or
$\chi^2_{MH}$ is not significant (on `adj_p`, so B and C items are always `flagged`), **C** if
$\lvert\Delta_{MH}\rvert\ge1.5$ and significantly greater than 1 by a one-sided test
($(\lvert\Delta_{MH}\rvert-1)/SE>z_{1-\alpha}$, 1.645 at `alpha=0.05`), **B** otherwise;
`classification="effect_size"` uses the thresholds 1 and 1.5 only (the codes difR prints).

`table` columns: `item`, `chi2`, `odds_ratio`, `log_odds_ratio`, `log_odds_ratio_se`, `delta`,
`delta_se`, `ets_class`, `p`, `adj_p`, `flagged`.

### `mantel`

```text
mantel(data, *, group, focal, items=None, anchor_items=None, matching=None, purify=False,
       max_iter=10, p_adjust="none", alpha=0.05) -> DIFResult
```

Mantel's (1963) test for ordered polytomous items and two groups: the sum of the focal group's
item scores against its expectation given the matching score ($df=1$; for 0 / 1 items it is
$\chi^2_{MH}$ without continuity correction). The effect size is the Liu & Agresti (1996)
cumulative common odds ratio, oriented like $\hat\alpha_{MH}$ (above 1 favours the reference
group; equal to $\hat\alpha_{MH}$ for 0 / 1 items), with Liu & Agresti's (1996, eq. 3) SE of
its log, which is valid both with a few large strata and with many small ones (as total-score
matching usually gives).

`table` columns: `item`, `chi2`, `df`, `cumulative_odds_ratio`, `log_cumulative_odds_ratio`,
`log_cumulative_odds_ratio_se`, `p`, `adj_p`, `flagged`.

### `generalized_mantel_haenszel`

```text
generalized_mantel_haenszel(data, *, group, focal, items=None, anchor_items=None,
                            matching=None, purify=False, max_iter=10, p_adjust="none",
                            alpha=0.05) -> DIFResult
```

Generalized Mantel–Haenszel test (Somes, 1986; Penfield, 2001) for several focal groups
(`focal` is a list) and / or unordered categories: in each stratum the groups x categories table
is compared with its expectation given the margins, $df=(G-1)(C-1)$ for $G$ groups and $C$
observed categories. With one focal group and 0 / 1 items it is $\chi^2_{MH}$ without continuity
correction.

`table` columns: `item`, `chi2`, `df`, `p`, `adj_p`, `flagged`.

### `logistic_dif`

```text
logistic_dif(data, *, group, focal, items=None, anchor_items=None, matching=None,
             purify=False, max_iter=10, p_adjust="none", alpha=0.05, kind="both",
             test="likelihood_ratio") -> DIFResult
```

Logistic-regression DIF for 0 / 1 items and two groups (Swaminathan & Rogers, 1990). With $S$
the matching score and $G$ the focal indicator, the models
$\operatorname{logit}P(x=1)=\beta_0+\beta_1S$ $(M_0)$, $+\beta_2G$ $(M_1)$ and $+\beta_3SG$
$(M_2)$ are fitted by maximum likelihood. `kind="both"` compares $M_0$ with $M_2$ ($df=2$),
`"uniform"` $M_0$ with $M_1$, `"non_uniform"` $M_1$ with $M_2$ ($df=1$); `test` is
`"likelihood_ratio"` or `"wald"` (on the added coefficients of the larger model). The effect
size is the difference of Nagelkerke's $R^2$ (Zumbo, 1999), with classes A / B / C by Zumbo &
Thomas (cut points 0.13, 0.26) and Jodoin & Gierl (0.035, 0.07) on $\Delta R^2$ alone. An
external `matching` may be continuous.

`table` columns: `item`, `chi2`, `df`, `delta_r2`, `zumbo_thomas_class`, `jodoin_gierl_class`,
`group_coefficient`, `group_coefficient_se`, `interaction_coefficient`,
`interaction_coefficient_se` (of $G$ and $SG$ in the larger model; NaN without such a term), `p`,
`adj_p`, `flagged`. A `ConvergenceWarning` is issued when a regression does not converge (for
example under separation).

### `DIFResult`

| Attribute | Type | Description |
|---|---|---|
| `table` | DataFrame | One row per tested item (columns above); `flagged` is `adj_p < alpha` |
| `flagged_items` | list[str] | Items flagged in `table` |
| `anchor_items` | list[str] | Items forming the matching score of the final step (empty with `matching`) |
| `n_iter`, `converged` | int, bool | Purification iterations and whether the flagged set repeated (`ConvergenceWarning` otherwise) |
| `flag_history` | DataFrame | Items flagged (unadjusted $p<$ `alpha`) at each step; row 0 is the first test |
| `method` | str | `"mantel_haenszel"`, `"mantel"`, `"generalized_mantel_haenszel"` or `"logistic"` |

---

## Exceptions and warnings

### `flexpyctt.errors`

Import with `from flexpyctt.errors import ...`.

#### `FlexpycttError`

Base class of every exception flexpyctt raises on purpose.

#### `SpecificationError`

The request is inconsistent with itself or with the data: an option outside its range, non-numeric
responses, an item with a single observed category and so on. Raised before any computation
starts. Subclass of `FlexpycttError` and `ValueError`.

#### `FlexpycttWarning`

Base class of every warning flexpyctt emits (subclass of `UserWarning`).

#### `ConvergenceWarning`

Estimation stopped without meeting the convergence criterion (e.g. `max_iter` reached).

#### `HeywoodWarning`

A boundary solution was reached (e.g. a correlation at 1 or a uniqueness at its bound).

#### `NotPositiveDefiniteWarning`

A correlation matrix (e.g. a pairwise polychoric matrix) was not positive definite and was smoothed
to a positive definite matrix.
