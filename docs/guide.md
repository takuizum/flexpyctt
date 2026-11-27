# User guide

English | [日本語](guide.ja.md)

flexpyctt is a package for item and test analysis that needs no fitted IRT model. This guide walks
through each function on simulated data: how to call it and how to read the result. Arguments and
return values are listed in the [API reference](reference.md). Fitting IRT models, and the analyses
that use a fitted model (local dependence, model-based parallel analysis, the dimensionality
report and so on), are in [flexpyirt](https://takuizum.github.io/flexpyirt/).

## 1. Setup

Pass a DataFrame (or array) with one row per respondent and one column per item. Responses are
ordered category codes; missing responses are NaN. Functions that draw random numbers take `rng`
(an integer seed or a `numpy.random.Generator`), so results are reproducible. There is no JAX and
no 64-bit setting.

The sections below use 12 binary items that measure one factor, answered by 1500 respondents.

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

## 2. Polychoric correlations

`flexpyctt.polychoric(data)` estimates the polychoric correlation matrix of ordinal responses
(tetrachoric for binary items), the usual input to eigenvalue-based checks of dimensionality. It
uses two-step maximum likelihood (Olsson, 1979): thresholds from each item's marginal proportions,
then one correlation per pair of items.

```python
from flexpyctt import polychoric

pc = polychoric(df)
print(pc.correlation.iloc[:4, :4].round(3))
print(pc.thresholds[items[0]].round(3))
print(pc.n_obs.iloc[:2, :2])
```

- Responses are ordered codes in any numbering (`1..5`, `0, 10, 20`, ...); only their order
  matters. Missing responses (NaN) use available cases: thresholds from every response to the
  item, each pair from respondents who answered both.
- A 2 x 2 table with exactly one empty cell has its maximum at $|\rho| = 1$. By default 0.5 is
  moved into the empty cell's diagonal, keeping the margins, as lavaan does
  (`zero_cell_correction=0` turns it off). Remaining maxima at $|\rho| \to 1$ give a
  `HeywoodWarning` and are listed in `pc.at_boundary`.
- A matrix assembled from pairwise estimates need not be positive definite. It is then smoothed
  (psych's `cor.smooth` rule) with a `NotPositiveDefiniteWarning`; `pc.unsmoothed_correlation`
  keeps the pairwise estimates.
- `flexpyirt.polychoric()` is a JAX implementation of the same estimator with the same
  arguments and result; the two agree to about $10^{-9}$.

## 3. Number of dimensions

Before fitting a unidimensional model it is worth asking how many dimensions the responses
support. Each method below is a diagnostic with known biases, so look at several of them together.

```python
from flexpyctt import efa, empirical_kaiser, hull, parallel_analysis, velicer_map

pa = parallel_analysis(df, rng=0, n_replications=20)
print(pa.n_factors, pa.observed[:3].round(3), pa.reference[:3].round(3))
print(velicer_map(pc).n_factors, empirical_kaiser(pc).n_factors)
print(hull(pc, rng=4).n_factors)
print(efa(pc, n_factors=1).loadings.head(3).round(3))
```

- `parallel_analysis()` (Horn, 1965) counts the leading eigenvalues of the polychoric matrix
  above the 95th percentile of eigenvalues from reference data without common factors. By
  default the reference permutes each item's responses independently, which keeps the item
  marginals; `reference="normal"` uses normal data instead, and `eigen_type="smc"` uses the
  reduced matrix. Results depend on `rng`; use more replications (the default is 100) for a
  final analysis.
- `velicer_map()` minimizes the average partial correlation after removing $m$ principal
  components; the revised fourth-power version is the default (`power=2` for the original).
- `hull()` fits factor models with $0, 1, \dots, J$ factors (`efa()`, ULS or ML) and picks the
  number after which more factors add little fit relative to the degrees of freedom they use
  (Lorenzo-Seva, Timmerman & Kiers, 2011); it needs at least 6 items.
- `comparison_data()` (Ruscio & Roche, 2012) generates populations with $1, 2, \dots$ factors that
  reproduce the observed correlations and item distributions, and adds a factor while it makes
  the eigenvalues of samples from that population closer to the observed ones.
- `empirical_kaiser()` compares each eigenvalue with a sample-size-dependent reference
  (Braeken & van Assen, 2017).
- These count dimensions; they do not test whether a unidimensional IRT model fits. For that,
  fit the model with flexpyirt and use its local dependence statistics, `FitResult.m2()` and
  `FitResult.item_fit()` ([flexpyirt user guide](https://takuizum.github.io/flexpyirt/guide/)).

## 4. DETECT and DIMTEST

DETECT (Zhang & Stout, 1999) looks at the covariance of item pairs among respondents with the
same total (or rest) score. Under essential unidimensionality these conditional covariances are
near zero; with several dimensions they are positive within clusters of items and negative
across them.

```python
from flexpyctt import detect, dimtest, exploratory_detect

halves = {item: ("A" if i < len(items) // 2 else "B") for i, item in enumerate(items)}
print(detect(df, halves).indices.round(3))
explore = exploratory_detect(df, max_clusters=3, rng=3)
print(explore.table.round(3))
test = dimtest(df, rng=5)
print(test.assessment_items, round(test.statistic, 3), round(test.p_value, 3))
```

- `detect()` scores a partition you supply (confirmatory DETECT); `exploratory_detect()`
  searches partitions on half of the respondents and reports them on both halves, so a
  partition that only fits the noise of the first half shows up as a low validation DETECT.
- A common reading: DETECT below 0.2 is essentially unidimensional, 0.2–0.4 weak, 0.4–1
  moderate and above 1 strong multidimensionality; ASSI above 0.25 or RATIO above 0.36 points to
  an approximately simple structure. For unidimensional data DETECT is close to zero or
  negative for every partition, as here.
- DETECT needs complete data. flexpyirt's `dimensionality()` includes the exploratory DETECT (validation half)
  among its unidimensionality checks.
- `dimtest()` (Stout, 1987; Stout, Froelich & Gao, 2001) turns the same idea into a test for
  binary items: it picks an assessment subtest on a third of the respondents and tests it on
  the rest. It is conservative, so a small $p$ is strong evidence of a second dimension and a
  large $p$ is only weak evidence for unidimensionality.

## 5. Classical item analysis

Before (or instead of) fitting an IRT model, classical item statistics show which items behave
badly.

```python
from flexpyctt import (
    coefficient_alpha,
    coefficient_omega,
    distractor_analysis,
    item_analysis,
    score_responses,
)

# Raw multiple-choice answers: "B" is correct; wrong answers are spread over A, C and D.
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

- `score_responses()` turns chosen options into 0 / 1 scores. Options are compared with `==`, so
  the key must use the data's codes and types (`1` does not match `"1"`; a key option that nobody
  chose gives a warning). A missing response scores 0, or stays NaN with `missing="keep"`.
- `item_analysis()` gives one row per item: `difficulty` (the mean divided by the largest
  score: the proportion correct for 0 / 1 items), `item_total` and `item_rest` (the correlation
  with the total score and with the total of the other items; point-biserial for 0 / 1 items) and
  `alpha_if_deleted`. Read `item_rest`: `item_total` includes the item itself and is inflated
  for short tests. `correlation="polyserial"` reports polyserial (biserial for 0 / 1 items)
  correlations instead, by the two-step estimator.
- `coefficient_alpha()` is KR-20 for 0 / 1 items. It also gives standardized alpha, Feldt's
  confidence interval and alpha with each item left out (`alpha.if_deleted`).
- `coefficient_omega()` fits one factor (ULS) to the item correlations and reports omega total.
  Like psych's `omega()`, it reverses items that load negatively, with a warning and the items in
  `reversed_items`; reverse-score such items yourself if the total you report keeps them as scored.
  With `correlation="polychoric"` it describes the latent responses behind the categories rather
  than the observed sum score.
- Missing responses (NaN) are used pairwise by default: every covariance or correlation from the
  respondents who answered both items. `missing="listwise"` uses complete rows only.
- `distractor_analysis()` has one row per item and option. A working distractor is chosen less
  often in higher score groups (`proportion_group_1` is the lowest third) and has a negative
  `point_biserial` with the rest score; the key should show the opposite. Here the wrong answers
  were drawn at random, so the distractors of an item look alike.

## 6. Observed-score DIF

flexpyctt screens items for DIF without fitting an IRT model: respondents are matched on their
total score (or on `anchor_items`, or an external `matching` variable) and the groups are compared
at each score. `mantel_haenszel()` gives the Mantel–Haenszel $\chi^2$, the common odds ratio
$\hat\alpha_{MH}$, the ETS delta $\Delta_{MH}=-2.35\ln\hat\alpha_{MH}$ and the ETS classes A / B / C;
`logistic_dif()` tests uniform and non-uniform DIF with nested logistic regressions and reports
Nagelkerke's $\Delta R^2$. `mantel()` (ordered polytomous items) and
`generalized_mantel_haenszel()` (several focal groups, or unordered categories) take the same
arguments. With `purify=True`, flagged items are removed from the matching score and the tests
are repeated until the flagged set repeats.

```python
from flexpyctt import logistic_dif, mantel_haenszel

# 20 binary items; the focal group has a lower mean, and items q03 and q07 are harder for it
# (uniform DIF).
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

- $\Delta_{MH}<0$ means the item is harder for the focal group at the same total score. Class C
  (here q03 and q07) needs $|\Delta_{MH}|\ge1.5$ and significantly above 1; class A means
  $|\Delta_{MH}|<1$ or not significant. `classification="effect_size"` uses the thresholds only.
- Matching is on the observed total, so with large group differences in ability and few items
  some items without DIF can be flagged (q17 here, with a negligible effect size): read the
  effect sizes, not only the p-values.
- The matching items must be complete: drop rows with missing responses first.

## 7. Errors and warnings

Every exception flexpyctt raises on purpose derives from `FlexpycttError`, and every warning from
`FlexpycttWarning` (`flexpyctt.errors`).

- `SpecificationError` (also a `ValueError`): the request is inconsistent with itself or with the
  data, e.g. an option outside its range, non-numeric responses, or an item with a single observed
  category. Raised before any computation starts.
- `ConvergenceWarning`: estimation stopped without meeting the convergence criterion (e.g.
  `max_iter` reached).
- `HeywoodWarning`: a boundary solution was reached (e.g. a correlation at 1 or a uniqueness at
  its bound).
- `NotPositiveDefiniteWarning`: a correlation matrix (e.g. a pairwise polychoric matrix) was not
  positive definite and was smoothed.

flexpyirt re-exports the same classes from `flexpyirt.errors`.
