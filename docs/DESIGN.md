# flexpyctt design

English | [日本語](DESIGN.ja.md)

The design of flexpyctt: what each tool computes, the formulas, the decisions and how the results
are checked. These sections were written in flexpyirt's design document while the tools lived
there, and moved here on 2026-11-27. A reference written "flexpyirt §x" points to
[flexpyirt's DESIGN.md](https://github.com/takuizum/flexpyirt/blob/develop/docs/dev/DESIGN.md); "V24" and the like are validation cases (§10), whose
numbers are shared with flexpyirt.

## 1. Scope and principles

- **Scope.** Item and test analysis that needs no fitted IRT model: polychoric correlations (§2),
  the number of dimensions (§3, §5, §6), DETECT (§4), DIMTEST (§7), observed-score DIF (§8) and
  classical item analysis (§9). A tool that needs a fitted model (a `FitResult`) belongs in
  flexpyirt: local dependence, the jackknife slope index, model-based parallel analysis and the
  `dimensionality()` ensemble are there (flexpyirt §16.11, §16.12, §16.17). The plan that these
  tools came from is flexpyirt §16.8.
- **No JAX.** flexpyctt imports NumPy, SciPy and pandas only, never JAX or flexpyirt, and has no
  64-bit setting to check (NumPy is float64). CI asserts that JAX is not installed.
- **Random numbers.** A function that draws random numbers takes `rng`, a
  `numpy.random.Generator` or an integer seed; `rng` is required where the result depends on it.
- **Errors and warnings.** `flexpyctt.errors` defines `FlexpycttError`, `SpecificationError`,
  `FlexpycttWarning`, `ConvergenceWarning`, `HeywoodWarning` and `NotPositiveDefiniteWarning`.
  flexpyirt re-exports the same class objects, so catching flexpyirt's classes also catches what
  a flexpyctt function raises (flexpyirt §16.18).
- **Compatibility with flexpyirt.** flexpyirt requires `flexpyctt>=0.X,<0.(X+1)` and re-exports
  the public names. A patch release keeps the public API and the four private helpers that
  flexpyirt imports (`correlation._coerce_ordinal`; `dimensionality._observed_correlation`,
  `_eigenvalues`, `_descending_eigenvalues`); a change to them needs a minor release
  (flexpyirt §16.24).
- **Modules.** `correlation` (`polychoric`, `PolychoricResult`, `smooth_correlation`),
  `_polychoric` (kernels), `dimensionality` (the tools of §3–§7 and their result classes), `_efa`,
  `_detect`, `_dimtest`, `_comparison_data`, `dif` (§8), `item_analysis` (§9), `errors`.

## 2. Polychoric correlations: the NumPy implementation

`polychoric()` implements the estimator defined in flexpyirt §16.9 (two-step maximum likelihood;
the same options, defaults, warnings and `PolychoricResult`). flexpyirt has a JAX implementation
of the same estimator; the two differ in how they compute $\Phi_2$ and the derivatives:

- $\Phi_2$ by Owen's (1956) $T$ function (`scipy.special.owens_t`):
  $\Phi_2(h,k;\rho)=\tfrac12\Phi(h)+\tfrac12\Phi(k)-T(h,a_h)-T(k,a_k)-\beta$ with
  $a_h=(k-\rho h)/(h\sqrt{1-\rho^2})$, $a_k=(h-\rho k)/(k\sqrt{1-\rho^2})$, $\beta=0$ if
  $hk>0$ or ($hk=0$ and $h+k\ge0$), else $\tfrac12$; $\Phi_2(0,0;\rho)=\tfrac14+
  \arcsin\rho/(2\pi)$; infinite limits are exact. This is independent of the JAX kernel's
  Genz (2004) quadrature, so the agreement of the two is a real check.
- Newton's method in $z=\operatorname{atanh}\rho$ for all pairs at once (array operations,
  no Python loop over pairs), with the analytic derivatives
  $\partial_\rho\pi_{ab}=\Delta\phi_2$ (Plackett, 1954) and
  $\partial_\rho\phi_2=\phi_2\,[\rho/s+(hk(1+\rho^2)-\rho(h^2+k^2))/s^2]$, $s=1-\rho^2$; the
  same step rule (Newton when concave, else a half step along the gradient; steps clipped to
  $\pm2$; step halving), bound $|z|\le8$, convergence criterion and boundary check as flexpyirt §16.9.

## 3. Parallel analysis, MAP and EKC (`parallel_analysis()`, `velicer_map()`, `empirical_kaiser()`; V25)

In `dimensionality.py` (NumPy for the $p\times p$ linear algebra and for the random reference
data, which is drawn from `rng`). These are D2 of the plan in flexpyirt §16.8. All three answer
question (A) of that plan (how many dimensions) and return frozen results with `n_factors`. Matrix inputs (DataFrame, array or
`PolychoricResult`) must be square, finite, symmetric, unit-diagonal, $p\ge3$.
**Counting**: leading eigenvalues strictly above the reference, stopping at the first miss
(psych's and EFAtools' sequential rule). **SMC eigenvalues**: $R$ with $h_j=1-1/(R^{-1})_{jj}$
on the diagonal (Humphreys & Ilgen, 1969); the EFA-communality variant of psych /
EFAtools waits for the EFA bridge (D5). **Parallel analysis references**: `"permutation"`
(default) permutes each item independently with `rng` (`(R, p, N)` permutations drawn at once by
`Generator.permuted`), so marginals and missingness are kept, then recomputes the correlation with
the validated `polychoric()` (or pairwise Pearson) per replication, warnings silenced; a
replication whose permuted pair is not estimable (missing data) is skipped and counted, with a
`FlexpyirtWarning`, and fewer than two usable replications raise `SpecificationError`.
`"normal"`: i.i.d. $N(0,1)$ data of the same $N\times p$ (rows of incomplete data count),
Pearson correlations, computed in chunks of replications. Quantiles use NumPy's default (linear interpolation, R's
type 7). Default quantile 0.95 (Glorfeld, 1995), PCA eigenvalues and the permutation
reference with polychoric correlations, following Garrido, Abad & Ponsoda (2013).
**MAP**: $f_m$ for $m=0,\dots,p-1$ with principal-component loadings
$V_m\Lambda_m^{1/2}$ (negative eigenvalues clipped at 0); power 4 is elementwise (O'Connor,
2000), and the default because the revised test is the recommended one (Velicer, Eaton &
Fava, 2000). The loop stops with NaN once a partial variance is $\le10^{-10}$ and the minimum
ignores NaN. EFA.dimensions' fourth-power statistic is a matrix power (documented, not
followed). **EKC**: Braeken & van Assen (2017) as in EFAtools (observed eigenvalues in the
remaining-variance term); with a `PolychoricResult` and no `n_obs`, the smallest pairwise
$n$ is used (conservative under missing data).
**Validation (V25)**: `validation/r/m5_dimensionality_reference.R` feeds the six lavaan
polychoric matrices of V24 to EFAtools 1.1.0 (`EKC`), psych 2.6.9 (`vss()$map`),
EFA.dimensions 0.1.9.3 (`MAP`) and an independent R implementation of both MAP powers, plus
`eigen()` for PCA / SMC eigenvalues: all agree to $10^{-11}$, including the selected numbers of
factors. Normal reference eigenvalues (3 sizes x PCA / SMC, 20000 data sets each) agree with
EFAtools' `PARALLEL` within Monte Carlo error ($|z|\le4.5$ for the means with a fixed key,
$6\times10^{-3}$ for the 95th percentiles). The permutation path is checked by properties
(independent permutations, recovery of two factors, reproducibility by key).

## 4. DETECT (`detect()`, `exploratory_detect()`; V27)

In `_detect.py` (NumPy). D4 of the plan in flexpyirt §16.8.
**Conditional covariances**: Zhang & Stout (1999): for each pair the ML covariance within
groups of equal total score $S$ and within groups of equal rest score $S-X_i-X_j$, each
averaged with the group proportions, then the two averaged (the total-score estimate is biased
negatively, the rest-score one positively). Groups are formed with `np.unique` and
`np.bincount`, one pass per pair. Raw item scores are used (polytomous items allowed);
complete data are required (the scores are undefined otherwise), so `dimensionality()`
reports DETECT as unavailable with missing data. **Indices** (sirt's `detect.index`):
DETECT $=100\,\overline{\delta\,\mathrm{ccov}}$, ASSI, RATIO, MADCOV100, MCOV100, unweighted
and weighted by $\sqrt{n_{ij}}$ (identical with complete data; kept for sirt's layout). The
smoothed (kernel) conditional covariances of sirt's default `conf.detect` are not
implemented. **Exploratory**: rows split into halves (`floor(N/2)` estimation rows drawn with
`rng`, or given), Ward clustering of the estimation-half matrix
$d=\max(\mathrm{ccov})-\mathrm{ccov}$ (the maximum includes the zero diagonal, as sirt's
`create.ccov`). R's `hclust(method = "ward.D")` applies the Lance–Williams Ward update to $d$;
SciPy's `"ward"` applies it to squared distances, so SciPy gets $\sqrt d$ (merge heights are
R's after squaring). `cutree` labels clusters by first appearance; flexpyirt relabels SciPy's
`fcluster` output the same way. Each partition $k=2..$`max_clusters` is scored on both halves;
`n_clusters` maximizes the estimation-half DETECT (sirt's choice). Zhang & Stout's genetic
partition search is not implemented. **Thresholds** (docstrings, guide): DETECT $<0.2$ / $0.4$
/ $1$, ASSI $0.25$, RATIO $0.36$ (Jang & Roussos, 2007; Zhang, 2007; flexpyirt §16.8 had 0.1 / 0.5 / 1,
corrected here). **Ensemble**: `dimensionality()` adds the validation-half DETECT of the selected
partition as a question-(B) row (supports when $<$ `detect_threshold` $=0.2$), available
without `fit`; the key is split three ways (parallel analysis, model-based PA, DETECT).
**Validation (V27)**: `validation/r/m5_detect_reference.R` (sirt 4.2.133): simulated
two-dimensional (true and wrong partitions), unidimensional and Science (polytomous) data with
`conf.detect(use_sum_score = TRUE, bias_corr = TRUE, smooth = FALSE)`, and a three-cluster data
set with `expl.detect(use_sum_score = TRUE, estsample = )`: conditional covariances,
all five indices (both weightings), partitions and both halves' indices agree to
$10^{-12}$ (observed $10^{-14}$), and Ward merge heights to $10^{-12}$. With the default
`smooth = TRUE`, sirt's `conf.detect(use_sum_score = TRUE)` standardizes the total score and then
subtracts raw item scores for the rest score, so its rest-score groups are wrong (DETECT 6.73
instead of 1.60 for `sim_2d`); recorded on the accuracy page, not followed.

## 5. EFA and the Hull method (`efa()`, `hull()`; V28)

In `_efa.py` (NumPy / SciPy). D5 of the plan in flexpyirt §16.8.
**Estimation**: unrotated solutions from the uniquenesses $\Psi$, minimized by SciPy L-BFGS-B
(`ftol=1e-15`, `gtol=1e-11`) from $\psi_i=1-\mathrm{SMC}_i$ within $[0.005,\max(1,\mathrm{SMC})]$
(psych / EFAtools' bounds). ULS: $F=\tfrac12\lVert R-\Psi-\Lambda\Lambda^\top\rVert_F^2$, with
$\Lambda$ the leading eigenvectors of $R-\Psi$ scaled by $\sqrt{\max(\lambda,0)}$, gradient
$\operatorname{diag}(\Lambda\Lambda^\top+\Psi-R)$ (psych's `FAgr.minres`, which is the exact
gradient of this $F$); this is psych's minres and EFAtools' ULS (psych's `fit.residuals` sums
only off-diagonal residuals, which has the same minimizer in the interior). ML: Jöreskog's
concentrated discrepancy on the eigenvalues of $\Psi^{-1/2}R\Psi^{-1/2}$ with gradient
$\operatorname{diag}(\Lambda\Lambda^\top+\Psi-R)/\psi^2$ (psych's `FAfn` / `FAgr`). Columns are
signed to non-negative sums; no rotation. A uniqueness at the floor is a Heywood case
(`HeywoodWarning` from `efa()`). The tolerances are tighter than R's `optim()` defaults, so the
flexpyirt objective is never above EFAtools' at its solution (checked in the tests).
**Fit indices** (EFAtools' `.gof`): CAF $=1-$KMO of the residual matrix with unit diagonal;
$\chi^2$ with Bartlett's multiplier $N-1-(2p+5)/6-2k/3$ and its $p$; CFI and RMSEA from the
uncorrected $F(N-1)$ and the null $-\log|R|(N-1)$, $df_0=p(p-1)/2$; $df=((p-k)^2-(p+k))/2$.
**Hull** (EFAtools' `HULL` / `.hull_calc`): rows $k=0..J$ with fit CAF, CFI or $1-$RMSEA
($k=0$: $1-$KMO($R$), 0, $1-$RMSEA$_0$; $df_0$); drop solutions whose fit does not increase,
remove interior points on or below the chord of their neighbours until none is, scree values
$st$ on the remaining interior, select the largest $st$ (fewer than three hull points: the best
fit, with a warning). Default $J$: one more than parallel analysis with SMC eigenvalues of normal
data and the mean rule (EFAtools' `efa_parallel` defaults, here 1000 replications drawn with `rng`),
capped at the largest $k$ with $df>0$ and raised to at least $\min(3,\text{cap})$; with
`max_factors` given, $J$ is taken as is. At least 6 variables. The table records `heywood` and
`converged` per solution, and a `HeywoodWarning` is emitted when the selected one is
inadmissible (as EFAtools). **Ensemble**: `dimensionality()` adds Hull (CAF, ULS, default $J$ from a
fourth split of the key) to question (A) when there are at least 6 items.
**Validation (V28)**: `validation/r/m5_efa_hull_reference.R` (EFAtools 1.1.0, psych 2.6.9) on the
V24 polychoric matrices with $\ge6$ variables and two simulated Pearson matrices (3 factors,
$N=500$; 2 weak factors, $N=300$): every $k\le J$ with ULS and ML — implied
$\Lambda\Lambda^\top$ and communalities to $10^{-4}$ (observed $\le7\times10^{-5}$, the largest at
Heywood solutions), CAF / CFI / RMSEA to $2\times10^{-4}$, $\chi^2$ to $10^{-4}$ relative, $df$
exactly; psych's `fa()` on the 3-factor matrix to $10^{-4}$. Hull for every matrix x estimator x
fit index with EFAtools' recorded $J$: the selected $k$ and the hull membership identical, the
fit values to $10^{-5}$ (observed $\le1.4\times10^{-6}$), $st$ to 1% relative (a ratio of small
differences; observed $\le3.5\times10^{-3}$ absolute). Comparison data is §6.

## 6. Comparison data (`comparison_data()`; V29)

In `_comparison_data.py` (NumPy; the random draws are passed in as arguments, so the generator
is deterministic given them; the public function draws them from `rng`). D5b of the plan in
flexpyirt §16.8.
For $k=1..K$ ($K$ = largest $k$ with positive $df$, or `max_factors`): resample each variable's
observed values `n_population` times and sort them; draw $F$ $(N,k)$ and $E$ $(N,p)$ standard
normal; generate the population (Ruscio & Kaczetow, 2008, as EFAtools'
`.simulate_cfm_empirical`): principal axis factoring of an intermediate $R^*$ (EFAtools'
`.paf_iter`: unit starting communalities, absolute eigenvalues, criterion $10^{-3}$ on the
largest change, `max_iter` 50), loadings clipped to $[-1,1]$ and flipped when the first is
negative, scores $F\Lambda^\top+E\,\mathrm{diag}(\sqrt{\max(1-h^2,0)})$, then for each column in
turn the rows are reordered by that column (stable sort) and the column replaced by its sorted
resampled values; $R^*$ starts at $R$ and moves by the residual $R-R_{\text{rep}}$ while the RMSE
of the lower-triangle residuals improves (from 1), otherwise halves the step from the best
$R^*$, stopping after 5 failures in a row. `n_samples` bootstrap samples of the data's size are
drawn from the population, and the RMSE of their Pearson-correlation eigenvalues against the
observed ones is recorded. $k$ is kept while the one-sided Mann–Whitney test (normal
approximation with continuity and tie corrections, as R's `wilcox.test` for $n\ge50$) says
the RMSE with $k$ factors is smaller than with $k-1$ at $\alpha=0.30$ (Ruscio & Roche's
recommendation); the result is the last kept $k$. Pearson correlations only (as Ruscio & Roche
and EFAtools); complete data. PAF loadings are signed to non-negative column sums: EFAtools
takes LAPACK's signs, which only matter for $k\ge2$ and do not change the population's
distribution. **Ensemble**: `dimensionality()` adds CD to question (A) with complete data
(fifth split of the key). **Validation (V29)**: `validation/r/m5_comparison_data_reference.R`
(EFAtools 1.1.0): PAF loadings for $k=1..3$ to $10^{-12}$ in $\Lambda\Lambda^\top$; the one-factor
population from EFAtools' own random draws (reproduced with the same seed and exported)
identical; the rank-sum $p$-values on EFAtools' RMSE columns to $10^{-10}$ relative; `CD()` on a
3-factor, a 2-factor ordinal and a 1-factor binary data set: the same suggestion for two keys
each and mean RMSEs within 15% (different random populations).

## 7. DIMTEST (`dimtest()`; V30)

In `_dimtest.py` (NumPy; random draws are arguments, and the public function draws them from
`rng`). D6 of the plan in flexpyirt §16.8. Binary complete data, at least 4 items. **Statistic** (Nandakumar & Stout,
1993): respondents grouped by the PT score; groups smaller than `min_group` (20) or with
$S_k\le0$ dropped; $T_L=K^{-1/2}\sum_k(\hat\sigma_k^2-\hat\sigma_{U,k}^2)/S_k$ with
$\hat\sigma_k^2$ the (divide-by-$J_k$) variance of the AT score, $\hat\sigma_{U,k}^2=\sum_i\hat
p_{ik}(1-\hat p_{ik})$ and $S_k^2=(\hat\mu_{4,k}-\hat\sigma_k^4+\hat\delta_{4,k})/J_k$,
$\hat\delta_{4,k}=\sum_i\hat p_{ik}(1-\hat p_{ik})(1-2\hat p_{ik})^2$. **Bias correction**
(Stout, Froelich & Gao, 2001): kernel-smoothed IRFs of the test sample (abilities
$\Phi^{-1}(r/(N+1))$ of total-score ranks with ties broken by uniform draws, Gaussian kernel
with bandwidth $1.1N^{-1/5}$ on 161 points of $[-4,4]$; Ramsay, 1991), $B$ (100) unidimensional
data sets of the test sample's size with $\theta\sim N(0,1)$ and IRFs interpolated linearly,
$T=(T_L-\bar T_G)/\sqrt{1+1/B}$, $p=1-\Phi(T)$. **AT selection**: with `assessment_items` the
whole sample is tested; otherwise a random third (`training_fraction`) selects the AT and the
rest is tested (Froelich & Habing, 2008, use a separate subsample): among the clusters of the
Ward clusterings into 2–6 groups of the training conditional covariances (§4), with 2 to
`max_assessment_size` (half the items) members, the one whose split {AT, rest} has the largest
DETECT. This is in the spirit of Froelich & Habing's conditional-covariance AT selection, not
their exact ATFIND, and is documented as such. **Ensemble**: `dimensionality()` adds DIMTEST
($p\ge\alpha$) to question (B) for complete binary data (sixth split of the key), otherwise an
unavailable row. **Validation (V30)**: no implementation outside the stand-alone DIMPACK, so the
test is accepted on `validation/simulation/dimtest_study.py` (results in
`dimtest_study.json`): 2PL data, slopes $U(0.8,1.8)$, intercepts $N(0,0.8)$, items in two halves
on dimensions correlated $\rho$, 200 replications per condition, defaults throughout:

| Data | $N$ | $J$ | $\rho$ | mean $T$ | SD $T$ | reject at .05 | reject at .10 |
|---|---|---|---|---|---|---|---|
| unidimensional | 1000 | 20 | 1 | -0.07 | 0.72 | 0.005 | 0.020 |
| unidimensional | 2000 | 20 | 1 | -0.12 | 0.71 | 0.010 | 0.030 |
| unidimensional | 2000 | 30 | 1 | -0.03 | 0.73 | 0.010 | 0.025 |
| two-dimensional | 1000 | 20 | 0.7 | 4.85 | 0.97 | 0.990 | 0.995 |
| two-dimensional | 2000 | 20 | 0.7 | 6.94 | 0.88 | 1.000 | 1.000 |
| two-dimensional | 2000 | 30 | 0.8 | 6.50 | 0.86 | 1.000 | 1.000 |

(Rerun on 2026-11-25 with flexpyctt's NumPy generator, flexpyirt §16.18. With the JAX keys used before
the move the rejection rates at .05 were 0.020, 0.005, 0.000 and 0.990, 1.000, 1.000.)

The test is conservative: $\bar T_G$ removes the bias of $T_L$ (mean $T$ near 0) but $T_L$'s
standard deviation under the null is about 0.72 rather than 1, so the type I error is below the
nominal level; power against two dimensions correlated 0.7–0.8 is 0.99–1. A small $p$ is
therefore strong evidence of multidimensionality, a large $p$ weak evidence for unidimensionality
(stated in the docstring and the guide). Standardizing by the bootstrap SD of $T_G$ instead would
recalibrate it but departs from the published statistic; not done. The pieces are unit-tested
against direct computations, and `tests/test_dimtest.py` reruns a small version of the
study (20 null and 10 alternative replications).

## 8. Observed-score DIF (`mantel_haenszel()`, `mantel()`, `generalized_mantel_haenszel()`, `logistic_dif()`, `DIFResult`; V47)

DIF screening without an IRT model, in `flexpyctt.dif` (public layer: input checks, matching
scores, purification, tables) and `flexpyctt._dif` (NumPy kernels for one studied item). The
API is the one approved on flexpyirt issue #102, with the changes listed at the end.

**Common arguments.** `data`, keyword-only `group` (one label per row), `focal` (one label;
a list for the generalized test; every other row is the reference group, as difR's
`focal.name`), `items=None` (tested items), `anchor_items=None` (items whose total is the
matching score), `matching=None` (external matching variable; not with `anchor_items` or
`purify`), `purify=False`, `max_iter=10`, `p_adjust="none"` (`"holm"`, `"BH"`; R's
`p.adjust` on the finite p-values), `alpha=0.05`. **Matching**: the total of the anchor
items, plus the studied item when it is not an anchor (difR's rule); every distinct value is
a stratum (no thick matching). **Missing values** on the tested or anchor items raise
`SpecificationError` (owner decision 3 of flexpyirt#102: error, not silent row deletion; difR's
`rowSums(na.rm = TRUE)` would mix totals over different item sets).

**Purification** (difR `purify = TRUE`, `puriadjType = "simple"`): step 0 tests every item
on the given anchors; while the flagged set (unadjusted $p<\alpha$) is non-empty, the next
step uses the anchors minus the flagged tested items; it stops when the flagged set equals
the previous one (`converged=True`) or after `max_iter` steps (`ConvergenceWarning`; the table
is the last step, whose anchors exclude the items flagged one step earlier, as difR). Nothing
flagged at step 0 gives `n_iter=0`, `converged=True`. With user `anchor_items`, purification
removes flagged items from that set only (difR ignores `purify` when `anchor` is given; this
is the natural extension, tested against a direct call with the final anchors). The final
`flagged` is `adj_p < alpha`. `flag_history` keeps difR's `difPur` (one row per step).

**Mantel–Haenszel** (Mantel & Haenszel, 1959; Holland & Thayer, 1988): strata with
$T_k>1$; $\chi^2_{MH}=(\lvert\sum A-\sum E(A)\rvert-c)^2/\sum\operatorname{Var}(A)$ with
$c=1/2$ under `continuity_correction=True` **always** (Holland & Thayer, difR; R's
`mantelhaen.test()` sets $c=0$ when the difference is below $1/2$, which changes the statistic
of items without DIF only); $\hat\alpha_{MH}=\sum AD/T/\sum BC/T$ with $A,B$ the reference
right / wrong counts (so $\hat\alpha_{MH}>1$ favours the reference group); Phillips & Holland
(1987) variance of $\ln\hat\alpha_{MH}$; $\Delta_{MH}=-2.35\ln\hat\alpha_{MH}$,
$SE(\Delta)=2.35\,SE(\ln\hat\alpha)$. **ETS classes** (owner decision 2: the full rule by
default; Zieky, 1993; Zwick, 2012): A if $\lvert\Delta\rvert<1$ or $\chi^2_{MH}$ is not
significant ($p\ge\alpha$ on the `p_adjust`-adjusted p-value, the one `flagged` uses, so B
and C items are always flagged; without adjustment it is the plain $p$); C if
$\lvert\Delta\rvert\ge1.5$ and $(\lvert\Delta\rvert-1)/SE(\Delta)>z_{1-\alpha}$ (a one-sided
test that $\lvert\Delta\rvert>1$, 1.645 at $\alpha=0.05$; some authors use 1.96); B otherwise. `classification="effect_size"`: A $<1\le$ B $<1.5\le$ C
(difR's printed codes use right-closed intervals; only exact ties differ). A zero or infinite
$\hat\alpha_{MH}$ gives a `FlexpycttWarning` (difR's "sparse data" warning).

**Mantel** (1963) for ordered items: the focal score sum against its hypergeometric
expectation, $df=1$, item values as scores; equal to $\chi^2_{MH}$ without correction for 0 / 1
items. **Liu–Agresti** (1996) cumulative common odds ratio
$\hat\psi=\sum_k\sum_{j<L}b_{jk}c_{jk}/T_k\big/\sum_k\sum_{j<L}a_{jk}d_{jk}/T_k$ ($a,c$: reference
/ focal counts at or below category $j$), **oriented as $\hat\alpha_{MH}$** (it reduces to
$\hat\alpha_{MH}$ for 0 / 1 items; difR's `liu_agresti_ccor()` reports the reciprocal). Its
SE is **Liu & Agresti's (1996) eq. 3**, which generalizes Robins, Breslow & Greenland (1986)
and is consistent both with a fixed number of large strata and with many sparse strata
("dually consistent"). With the focal group as their row 1,
$\widehat{\operatorname{Var}}(\ln\hat\psi)=\sum_k\hat\xi_k\big/(\sum_k\sum_jb_{jk}c_{jk}/T_k)^2$,
$\hat\xi_k=\sum_j\hat\phi_{jjk}+2\sum_{j<s}\hat\phi_{jsk}$,
$\hat\phi_{jsk}=\frac{n_{Fk}n_{Rk}}{T_k^2}\{\hat\psi d_{sk}a_{jk}/n_{Fk}\,[1+(\hat\psi-1)a_{sk}/n_{Rk}]
+c_{jk}b_{sk}/n_{Rk}\,[\hat\psi-(\hat\psi-1)c_{sk}/n_{Fk}]\}$ ($j\le s$). Matching on a
total score gives many small strata, where a large-strata delta method
($\operatorname{Var}(S-\hat\psi R)/S^2$ with plug-in multinomial variances, used before
review) is far too small: about 40% at four respondents per stratum. The estimator is not
invariant to reversing the category order; Liu & Agresti note that one can average the two
orderings and that the difference is trivial in practice. flexpyctt uses the categories in
increasing order as published, without averaging. difR's own SE is wrong (VALIDATION V47).

**Generalized MH** (Landis, Heyman & Koch, 1978; Somes, 1986; Penfield, 2001): per stratum
($T_k>1$) the $G\times C$ table of groups x observed categories; observed focal counts for
categories $1..C-1$ against $T\,\mathbf p\otimes\mathbf q$, covariance
$T^2/(T-1)\,(D_p-pp^\top)\otimes(D_q-qq^\top)$ restricted to those cells; $df=(G-1)(C-1)$.
A singular summed covariance (a category or group absent from every informative stratum) is
handled with the pseudo-inverse and $df=$ rank, with a `FlexpycttWarning`. It serves both
several focal groups with 0 / 1 items (difR `difGMH()`) and nominal categories
(`mantelhaen.test()` on an $I\times J\times K$ table).

**Logistic regression** (Swaminathan & Rogers, 1990; Zumbo, 1999): models
$S$, $S+G$, $S+G+SG$ fitted by Newton–Raphson in NumPy (no new dependency), from
$\beta=0$, until the max relative step is below $10^{-12}$ (100 iterations at most; a
singular information matrix, as under separation, stops it with a `ConvergenceWarning` and a
pseudo-inverse covariance). LRT = difference of deviances; Wald on the added coefficients of
the larger model (difR's `criterion = "Wald"`). Nagelkerke $R^2$ with $N$ = rows, $\Delta R^2$
= larger minus smaller model. Zumbo–Thomas (0.13, 0.26) and Jodoin–Gierl (0.035, 0.07)
classes on $\Delta R^2$ only, as difR prints them (the published rules add significance,
which is the `flagged` column); unlike the ETS default, no significance is folded in because
the cut points were proposed as effect sizes to read next to the test.

**Changes from the API proposed on flexpyirt#102.** (1) `mantel()`: `cumulative_odds_ratio_se` became
`log_cumulative_odds_ratio_se`, next to a new `log_cumulative_odds_ratio`, because the SE is of
the log (as `log_odds_ratio_se` of `mantel_haenszel()`). (2) `mantel()`, `generalized_mantel_haenszel()`
and `logistic_dif()` have a `df` column (also for the fixed-$df$ methods, so tables of
different methods line up). (3) `DIFResult` also has `flag_history`, `method` and the property
`flagged_items`. (4) The logistic coefficient columns are named `group_coefficient`,
`group_coefficient_se`, `interaction_coefficient`, `interaction_coefficient_se`. (5) Owner
decision 4 (standardization, STD P-DIF): not included, as in the scope list of flexpyirt#102.
**Not in this PR**: ordinal logistic DIF (the second PR of flexpyirt#102).

## 9. Classical item analysis (V46)

In `item_analysis.py` (NumPy / SciPy / pandas), re-exported by flexpyirt (flexpyirt issue #101). The API is the one proposed in the issue and approved
by the owner (2026-11-26): `score_responses()`, `item_analysis()`, `coefficient_alpha()`
(`AlphaResult`), `coefficient_omega()` (`OmegaResult`), `distractor_analysis()`. The names do not
clash with flexpyirt's model-based `reliability()`. **Changes from the issue's sketch** (no
signature changed): `AlphaResult` also carries `confidence_level` and `if_deleted` has both
`alpha` and `standardized_alpha`; `OmegaResult` also carries the `correlation` matrix it was
fitted to and the `reversed_items` reversed before computing omega; `score_responses()` also
accepts the key as a sequence in column order.
**Missing data** (owner decision 2: the issue's `"pairwise"` default): everything is computed
from the pairwise-complete covariance matrix $C$ (R's `cov(use = "pairwise")`) and, for
standardized alpha and omega, the pairwise-complete correlation matrix (R's
`cor(use = "pairwise")`, not `cov2cor(C)`), as psych's `alpha()`. `"listwise"` drops incomplete
rows first. Rows with no response are dropped in both cases, so `n_obs` (the Feldt $n$) counts
respondents with at least one response; psych uses `nrow(x)`.
**Item–total and item–rest correlations** are $c_j/\sqrt{C_{jj}\,\mathbf 1^\top C\mathbf 1}$ and
$(c_j-C_{jj})/\sqrt{C_{jj}(\mathbf 1^\top C\mathbf 1-2c_j+C_{jj})}$ with $c_j=\sum_l C_{jl}$,
the Pearson correlations with $T$ and $T-x_j$ on complete data. psych's `r.drop` is the same
formula; its `raw.r` correlates with `rowSums(x, na.rm = TRUE)`, which treats a missing
response as 0 in the total and so is not the quantity alpha is built from. flexpyctt keeps the
$C$-based definition for both columns so that they, `alpha` and `alpha_if_deleted` describe
the same matrix; the difference from `raw.r` under missing data is tested and documented.
**Polyserial / biserial** (`correlation="polyserial"`): the two-step estimator of Olsson,
Drasgow & Dorans (1982), $\hat\rho=r\,\sigma_j/\sum_k\phi(\tau_{jk})$, with $\sigma_j$ the
item's SD (denominator $n$) and $\tau_{jk}=\Phi^{-1}$ of its cumulative proportions, both over
the item's observed scores; $r$ is the Pearson item–total (–rest) correlation above. This is
polycor's `polyserial(ML = FALSE)`, psych's `polyserial()` on complete data and, for two
categories, $r\sqrt{pq}/\phi(\Phi^{-1}(p))$. The issue's "thresholds from the margins, then
$\rho$" means this estimator: polycor's two-step version does not maximize over $\rho$ (the
maximum-likelihood polyserial, `ML = TRUE`, is not provided). psych's `biserial()` divides by
the $n-1$ SD of the total while using $p,q$, so it is this value times $\sqrt{(n-1)/n}$
(tested). Estimates outside $[-1,1]$ are set to $\pm1$ with a `HeywoodWarning`, as psych does.
`difficulty` is the mean over the largest observed score, so scores must be $\ge0$.
**Omega**: $\omega_t=1-\sum_j(1-\lambda_j^2)/\mathbf s^\top R\mathbf s$ from the one-factor ULS
solution (`_efa.fit_factors`, the kernel of `efa()`), with $s_j=\operatorname{sign}(\lambda_j)$
($+1$ for a zero loading), i.e. psych's `omega.tot` (Revelle & Zinbarg, 2009) with its default
`flip = TRUE` and the observed total variance in the denominator. Items that load negatively
are reversed, as psych does (review of flexpyirt#127): otherwise one reverse-keyed item makes
$\mathbf 1^\top R\mathbf 1$ small and omega describes a total nobody would use. The reversal is
not silent: a `FlexpycttWarning` names the items and `OmegaResult.reversed_items` lists them,
while `loadings` and `correlation` stay those of the items as scored; reverse-scoring the items
first gives the same omega (tested). The model-implied form
$(\sum\lambda)^2/((\sum\lambda)^2+\sum\psi)$ equals it only under perfect fit and is not
reported. Calling the kernel instead of `efa()` admits three items (a just-identified model,
which `efa()` rejects for having no degrees of freedom). The Pearson matrix is smoothed with
`smooth_correlation` when not positive definite (pairwise matrices may not be), as
`polychoric()` does. lavaan's one-factor CFA (listed in the issue) was not used as a reference:
it fits ML to the covariance matrix, a different estimator on a different scale.
**Keyed scoring and options**: options are compared with `==` per cell, so codes of any type
work; a key option never chosen warns (usually a key in the wrong codes or type). In
`distractor_analysis()` a missing response scores 0 in $T$; each option's statistics use the
item's respondents; the key is listed even if nobody chose it. Score groups follow R's
`cut(T, quantile(T, 0:G/G), include.lowest = TRUE)` (type-7 quantiles, ties never split); equal
quantiles raise instead of silently merging groups.
**Validation (V46)**: `validation/r/ctt_item_analysis_reference.R`, see
§10. **Left out** (as in the issue): hierarchical $\omega$, split-half, Guttman
$\lambda$, glb, the upper–lower 27% index, plots (flexpyirt#103), weights.

## 10. Validation

Reference values come from R and are committed as JSON in `tests/reference/`; CI does not run R.
The generator scripts are in `validation/r/` (run them from the repository root), and
`validation/simulation/` holds the DIMTEST study. The case numbers continue flexpyirt's list
(V1–V57), so that a case keeps its number in both repositories.

| ID | Contents | Reference | MS |
|---|---|---|---|
| V24 | Polychoric / tetrachoric correlations (simulated tetrachoric, mixed 2–5 categories, 10% MCAR, one-empty-cell 2 x 2 tables, LSAT7, Science) and smoothing | lavaan `lavCor(..., ordered = , missing = )`; psych `polychoric(..., smooth = FALSE, correct = 0)`, `cor.smooth()` | M5 |
| V25 | Number of dimensions: EKC, MAP (squared and fourth power), PCA / SMC eigenvalues on the V24 matrices; normal-data parallel-analysis references (3 sizes) | EFAtools `EKC()`, `PARALLEL()`; psych `vss()$map`; EFA.dimensions `MAP()`; independent R fourth-power MAP | M5 |
| V27 | DETECT: confirmatory (simulated 2D with true / wrong partitions, 1D, Science) and exploratory (simulated 3 clusters, given split) | sirt `conf.detect(use_sum_score = TRUE, smooth = FALSE)`, `expl.detect(use_sum_score = TRUE, estsample = )` | M5 |
| V28 | EFA (ULS, ML; every $k$ up to Hull's bound) and the Hull method (CAF, CFI, RMSEA) on the V24 matrices with $\ge 6$ variables and two simulated matrices | EFAtools `EFA(method = , rotation = "none")`, `HULL(method = , gof = )`; psych `fa(fm = "minres" / "ml", rotate = "none")` | M5 |
| V29 | Comparison data: PAF, the one-factor population generator from exported draws, rank-sum tests, `CD()` on 3-factor continuous, 2-factor ordinal and 1-factor binary data | EFAtools `.paf_iter`, `.simulate_cfm_empirical`, `CD()`; `wilcox.test(x, y, "less")` | M5 |
| V30 | DIMTEST: type I error and power (2PL; $N$ 1000–2000, $J$ 20–30; unidimensional and two dimensions correlated 0.7–0.8; 200 replications each) | none (no implementation outside DIMPACK): `validation/simulation/dimtest_study.py` | M5 |
| V46 | Classical item analysis (flexpyctt): keyed scoring, alpha (pairwise and listwise), item statistics, point-biserial / biserial / polyserial correlations, omega total (Pearson and polychoric), option analysis; simulated 10-item multiple-choice (complete, 5% MCAR) and 6-item 0–4 data (complete, 6% MCAR) | none (not mirt): psych `alpha()`, `score.multiple.choice()`, `responseFrequency()`, `polyserial()`, `biserial()`, `fa()`, `omega(nfactors = 1)`; ltm `cronbach.alpha()`, `biserial.cor()`; polycor `polyserial(ML = FALSE)`; lavaan `lavCor()`; base R for option statistics | flexpyctt |
| V47 | Observed-score DIF (flexpyctt, flexpyirt#102): Mantel–Haenszel with and without continuity correction, Holm / BH, purification, user anchors, an external matching variable (`difR::verbal`, simulated 2PL with uniform and non-uniform DIF); generalized MH for three groups and for nominal categories (simulated graded responses); Mantel's test and the Liu–Agresti odds ratio; logistic regression (both / uniform / non-uniform, LRT / Wald, purification, anchors, matching) | difR 6.1.0 `difMH()`, `difGMH()`, `difLogistic()`, `difMantel.poly()`, `liu_agresti_ccor()`; `stats::mantelhaen.test()`; direct R implementations: `validation/r/ctt_dif_reference.R`; the worked example of Liu & Agresti (1996, Table 1) | — |

Notes on each case:

- **Polychoric correlations (V24, M5)**: not a mirt comparison. lavaan's `lavCor` on ordered
  variables (two-step ML by `nlminb` with analytic gradient and Hessian, $|\rho|\le0.999$, default
  `zero.add = c(0.5, 0)`) is the tight reference; a lavaan warning about non-convergence stops the
  script. psych's `polychoric` uses `optimize()` at its default tolerance and `mnormt::sadmvn`, so
  it is checked to 1e-3 only, on complete data and excluding pairs with a corrected zero cell
  (psych with `global = TRUE` applies no correction); with unequal category counts psych takes
  per-pair thresholds, which differs under missing data (6e-3 on the MCAR case), so that case is
  lavaan-only. A further self-check confirms that each pair's two-step gradient vanishes at the
  estimate. The bivariate normal CDF is unit-tested against Plackett's integral.
- **Number of dimensions (V25, M5)**: deterministic quantities are computed in R from the V24
  lavaan matrices, so both sides see the same input. EFA.dimensions' `PARALLEL()` masks
  EFAtools' when both are attached; the script calls `EFAtools::` explicitly. EFA.dimensions'
  fourth-power MAP is $(\operatorname{tr}(R^{*4})-p)/(p(p-1))$ (a matrix power; it exceeds 1 on
  these data), so the fourth power is compared with an independent elementwise implementation
  and EFA.dimensions' values are only recorded. Normal-data references are stochastic on both
  sides; the test uses a fixed key and compares the means by $z$ (standard error from the
  replications, $\sqrt2$ for two independent means) and the 95th percentiles in absolute value.
- **DETECT (V27, M5)**: sirt's `conf.detect()` is called with `smooth = FALSE`; with the
  default `smooth = TRUE` and `use_sum_score = TRUE` it subtracts raw item scores from the
  standardized total score, which mis-forms the rest-score groups (DESIGN §4). `expl.detect()`
  forces `smooth = FALSE` itself. Its returned `detect.unweighted` holds only the validation-half
  indices, so the script recomputes the estimation-half indices with `conf.detect()` on the
  estimation rows.
- **EFA and Hull (V28, M5)**: unrotated loadings are compared through the implied
  $\Lambda\Lambda^\top$ (invariant to column signs and to the order of equal eigenvalues).
  `HULL()`'s bound $J$ comes from a random parallel analysis, so the script records
  `settings$n_fac_max` and the test passes it as `max_factors`. EFAtools' `optim()`-based solutions
  are less tightly converged than flexpyirt's, so the test also checks that flexpyirt's objective is
  not above EFAtools' at EFAtools' $\psi=1-h^2$; the remaining differences are largest at Heywood
  solutions (bounded uniquenesses), which Hull includes for $k$ beyond the true number of factors.
- **Comparison data (V29, M5)**: CD is random on both sides. The generator is checked
  exactly by reproducing EFAtools' random draws in R (`set.seed`, then the same `sample()` /
  `rnorm()` calls in the generator's order), exporting them and feeding them to flexpyirt; this is
  exact for one factor, where the loadings' sign is fixed by the generator. The selection is
  checked on data sets where the answer is clear; the mean RMSEs are compared loosely because
  each side draws its own populations: 15% for the retained numbers of factors, and 60% for the
  rejected extra factor, whose single population varies much more (0.044 to 0.072 over six seeds
  where EFAtools' draw gave 0.048). JSON `auto_unbox` would turn a one-element $p$-value vector
  into a scalar, so the script writes `as.list()`.
- **DIMTEST (V30, M5)**: accepted on a simulation study instead of a reference
  comparison (DESIGN §7 has the table): with 200 replications the type I error at .05 is
  0.5–1.0% (conservative; the bootstrap removes the bias but $T$'s SD under the null is about 0.72)
  and the power 0.99–1.00. Rerun with `uv run python validation/simulation/dimtest_study.py`
  (about 4 minutes); the tests run 20 null and 10 alternative replications.
- **Classical item analysis (V46, flexpyctt, flexpyirt#101)**:
  `validation/r/ctt_item_analysis_reference.R` (R 4.6.2, psych 2.6.9, ltm
  1.2.0, polycor 0.8.2, lavaan 0.7.3) writes `tests/reference/v46_item_analysis.json`;
  `tests/test_item_analysis.py` compares. Four simulated cases: 10
  multiple-choice items ($N=500$, keys and distractors of varying pull, one item with five
  options) complete and with 5% MCAR, and 6 items scored 0–4 ($N=300$, graded model) complete and
  with 6% MCAR; plus three items for a just-identified omega and the complete 0–4 data with one
  item reverse-scored, where psych's default `flip = TRUE` reverses it (its `flip = FALSE` value
  is also recorded, 0.18 lower, and equals the unreversed formula). Observed differences: closed forms
  (alpha, standardized alpha, if-deleted values, Feldt bounds, means, SDs, item–total and
  item–rest correlations, point-biserial, polyserial and biserial, option proportions,
  point-biserials, mean scores and group proportions) $\le6\times10^{-15}$; keyed scores
  identical; omega total $\le7\times10^{-8}$ and loadings $\le1.4\times10^{-6}$ (psych's `fa()`
  stops its minres optimization earlier); polychoric matrices against lavaan $\le2\times10^{-8}$; the reversed item is the one psych
  reverses, and omega with it reversed equals omega of the data reverse-scored by hand.
  Known differences, each asserted in the tests: (1) psych's `raw.r` correlates with
  `rowSums(na.rm = TRUE)`, so with missing data it is compared with the $C$-based
  `colSums(C)/sqrt(diag(C) * sum(C))` computed in R, and `raw.r` itself only on complete or
  listwise data (DESIGN §9); `r.drop` is $C$-based in psych too and is compared in every
  case. (2) ltm's raw `cronbach.alpha(na.rm = TRUE)` takes the total variance from complete rows
  and the item variances from all observed scores, which is neither rule; it is compared on
  complete data only (its standardized alpha uses complete rows and matches `"listwise"`).
  (3) psych's `biserial()` equals the two-step biserial times $\sqrt{(n-1)/n}$. (4) psych's
  `polyserial()` counts missing rows in its $n$, so it is compared on complete data; with missing
  data the two-step factor $\sigma_j/\sum_k\phi(\tau_{jk})$ is recomputed in R. (5) Option
  statistics have no installed R implementation (TAM's `tam.ctt3` is not installed); the script
  computes them in base R from their definitions, and option proportions also match psych's
  `responseFrequency()`. lavaan's one-factor CFA, listed in the issue, was not used (ML on the
  covariance matrix is another estimator and scale).
- **Observed-score DIF (V47, flexpyctt, flexpyirt#102)**: references from difR 6.1.0 and base R,
  committed in `tests/reference/v47_dif.json`. Measured agreement: MH
  $\chi^2$, $\hat\alpha_{MH}$ and Phillips–Holland variance, GMH statistics (difGMH and
  `mantelhaen.test()` on groups x categories x strata tables), Mantel statistics and odds ratios
  to $10^{-10}$ relative (the computations are closed form); purification paths (`difPur`),
  iteration counts, convergence flags and final flags equal. Logistic regression: `difLogistic()`
  calls `glm()` with its default `epsilon = 1e-8`, which leaves the coefficients and statistics
  up to $2\times10^{-4}$ (relative) from the MLE on these data, so the script replaces
  `stats::glm.control` by one with `epsilon = 1e-14`; then LRT statistics, $\Delta R^2$ and
  coefficients agree to $10^{-8}$ and SEs and Wald statistics to $10^{-6}$ (`glm()`'s `vcov()`
  uses the IRLS weights of the previous iterate; the deviance criterion stops at a step of about
  $\sqrt{\epsilon}$). Known differences, checked in the tests: (1) `mantelhaen.test()` drops the
  continuity correction when $\lvert\sum A-\sum E(A)\rvert<1/2$ (difR and flexpyctt do not; three
  items of the simulated set); (2) `difMantel.poly()` rounds its output (statistic to 3, odds
  ratio to 4 decimals) and its purification removes a flagged item from its own matching
  score, so Mantel's statistic and its purification path are recomputed in R with difR's rule
  for the other methods, and `difMantel.poly()` is a rounded cross-check; (3)
  `liu_agresti_ccor(se = "liu_agresti")` differentiates $T=\sum ad/N-\hat\psi\sum bc/N$ with
  respect to the focal cumulative count as $a/N-\hat\psi b/N$ (the sign of $a/N$ is wrong) and
  divides by $\sum bc/N$ instead of $\sum ad/N$, so its `SE_log_Psi` is neither
  $SE(\ln\hat\psi)$ nor $SE(\hat\psi)$ (0.031 against a Monte Carlo SD of 0.0296 for $\psi=1.48$,
  300 replications of $N=20000$ in three strata; 0.14 against the Phillips–Holland 0.36 on a
  `verbal` item). flexpyctt and the R script's `la_eq3()` use instead Liu & Agresti's (1996)
  eq. 3, which stays valid with many small strata (DESIGN §8). Its R implementation, written
  in the paper's notation and independently of the Python one, reproduces the paper's worked
  example (Table 1, 28 centres: $\ln\hat\theta=-1.153$, SE .571), and the Python tests check
  the same published values, Mantel's 4.84 included. A plug-in delta method (used before
  review; 0.0291 against the Monte Carlo 0.0296 with large strata) gives 0.375 there and is about
  40% too small with four respondents per stratum. A Python Monte Carlo test (150 strata of 4
  and of 10 respondents, a four-category proportional-odds item, 400 replications) checks that
  the mean eq. 3 SE is within 10% of the Monte Carlo SD of $\ln\hat\psi$, and another checks
  that for 0 / 1 items with large strata it is within 1% of the Phillips–Holland SE. (4) `liu_agresti_ccor()` reports $1/\hat\psi$ of
  flexpyctt (difR's orientation: above 1 favours the focal group).
