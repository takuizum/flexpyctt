# flexpyctt 設計

[English](DESIGN.md) | 日本語

flexpyctt の設計: 各道具が何を計算するか、数式、決めたこと、結果の確かめ方。これらの節は、道具が flexpyirt にあった
ときに flexpyirt の設計文書に書かれ、2026-11-27 にここへ移した。「flexpyirt §x」と書いた参照は
[flexpyirt の DESIGN.ja.md](https://github.com/takuizum/flexpyirt/blob/develop/docs/dev/DESIGN.ja.md) を指す。「V24」などは照合ケース（§10）で、番号は flexpyirt と
共通である。

## 1. 範囲と原則

- **範囲。** 推定済みの IRT モデルを使わない項目・テスト分析: ポリコリック相関（§2）、次元数（§3、§5、§6）、
  DETECT（§4）、DIMTEST（§7）、観測得点による DIF（§8）、古典的項目分析（§9）。推定済みのモデル（`FitResult`）が
  要る道具は flexpyirt に置く。局所依存、ジャックナイフ傾き指標、モデルに基づく平行分析、`dimensionality()` の
  アンサンブルは flexpyirt にある（flexpyirt §16.11、§16.12、§16.17）。これらの道具のもとになった計画は
  flexpyirt §16.8 にある。
- **JAX を使わない。** flexpyctt は NumPy・SciPy・pandas だけを import し、JAX も flexpyirt も import しない。
  64 ビットの設定を確認する必要もない（NumPy は float64）。CI で、JAX がインストールされていないことを確認する。
- **乱数。** 乱数を使う関数は `rng`（`numpy.random.Generator` または整数のシード）を取る。結果が乱数に依存する
  関数では `rng` は必須である。
- **エラーと警告。** `flexpyctt.errors` が `FlexpycttError`、`SpecificationError`、`FlexpycttWarning`、
  `ConvergenceWarning`、`HeywoodWarning`、`NotPositiveDefiniteWarning` を定義する。flexpyirt は同じクラス
  オブジェクトを再公開するので、flexpyirt のクラスで捕まえれば、flexpyctt の関数が送出したものも捕まる
  （flexpyirt §16.18）。
- **flexpyirt との互換性。** flexpyirt は `flexpyctt>=0.X,<0.(X+1)` を要求し、公開名を再公開する。パッチリリースは
  公開 API と、flexpyirt が import している非公開のヘルパー 4 つ（`correlation._coerce_ordinal`、
  `dimensionality._observed_correlation`、`_eigenvalues`、`_descending_eigenvalues`）を変えない。変える場合は
  マイナーリリースにする（flexpyirt §16.24）。
- **モジュール。** `correlation`（`polychoric`、`PolychoricResult`、`smooth_correlation`）、`_polychoric`
  （カーネル）、`dimensionality`（§3〜§7 の道具とその結果クラス）、`_efa`、`_detect`、`_dimtest`、
  `_comparison_data`、`dif`（§8）、`item_analysis`（§9）、`errors`。

## 2. ポリコリック相関: NumPy による実装

`polychoric()` は、flexpyirt §16.9 で定義した推定量（2 段階の最尤法。オプション、既定値、警告、`PolychoricResult` は
同じ）を実装する。flexpyirt には同じ推定量の JAX による実装があり、2 つは $\Phi_2$ と微分の計算方法が異なる:

- $\Phi_2$ は Owen (1956) の $T$ 関数（`scipy.special.owens_t`）で計算する:
  $\Phi_2(h,k;\rho)=\tfrac12\Phi(h)+\tfrac12\Phi(k)-T(h,a_h)-T(k,a_k)-\beta$、
  $a_h=(k-\rho h)/(h\sqrt{1-\rho^2})$、$a_k=(h-\rho k)/(k\sqrt{1-\rho^2})$、$\beta$ は $hk>0$ または
  （$hk=0$ かつ $h+k\ge0$）のとき 0、それ以外は $\tfrac12$。$\Phi_2(0,0;\rho)=\tfrac14+\arcsin\rho/(2\pi)$。
  無限大の限界は正確に扱う。JAX 版の Genz (2004) の求積とは独立な計算なので、2 つの一致は意味のある確認になる。
- $z=\operatorname{atanh}\rho$ についての Newton 法を、すべてのペアについて同時に行う（配列演算。ペアについての
  Python ループは無い）。微分は解析的: $\partial_\rho\pi_{ab}=\Delta\phi_2$（Plackett, 1954）、
  $\partial_\rho\phi_2=\phi_2\,[\rho/s+(hk(1+\rho^2)-\rho(h^2+k^2))/s^2]$、$s=1-\rho^2$。ステップの規則（凹なら
  Newton、そうでなければ勾配方向に 0.5。ステップは $\pm2$ に制限。ステップ半減）、境界 $|z|\le8$、収束判定、境界解の
  判定は flexpyirt §16.9 と同じ。

## 3. 平行分析・MAP・EKC（`parallel_analysis()`、`velicer_map()`、`empirical_kaiser()`。V25）

`dimensionality.py` に置く（$p\times p$ の線形代数も、`rng` から引く参照データの乱数も NumPy）。flexpyirt §16.8 の
計画の D2 にあたる。3 つとも、その計画の問い (A)（次元はいくつか）に答え、`n_factors` をもつ変更不可の結果を返す。行列の入力（DataFrame、配列、`PolychoricResult`）は
正方・有限・対称・対角 1・$p\ge3$ であること。**数え方**: 参照を厳密に上回る固有値を先頭から数え、最初に下回った
ところで止める（psych・EFAtools と同じ逐次の規則）。**SMC の固有値**: 対角を $h_j=1-1/(R^{-1})_{jj}$ にした $R$
（Humphreys & Ilgen, 1969）。psych・EFAtools の EFA の共通性を使う版は EFA 連携（D5）を待つ。**平行分析の参照**:
`"permutation"`（既定）は `rng` で各項目を独立に並べ替え（`Generator.permuted` で `(R, p, N)` の置換を一度に引く）、
周辺分布と欠測を保ったまま、反復ごとに照合済みの `polychoric()`（またはペアごとの Pearson）で相関を求め直す
（警告は抑える）。並べ替えたペアが推定できない反復（欠測による）は飛ばして数え、`FlexpyirtWarning` を出す。
使える反復が 2 つ未満なら `SpecificationError`。`"normal"` は同じ $N\times p$ の独立な $N(0,1)$ データ（欠測のある
行も数える）の Pearson 相関で、反復をいくつかずつに分けて計算する。分位点は NumPy の既定（線形補間、R の type 7）。既定の
分位点 0.95（Glorfeld, 1995）、PCA の固有値、ポリコリック相関での並べ替えの参照は Garrido, Abad & Ponsoda (2013)
に従う。**MAP**: 主成分の負荷量 $V_m\Lambda_m^{1/2}$（負の固有値は 0 に切る）で $m=0,\dots,p-1$ の $f_m$ を
求める。4 乗は要素ごと（O'Connor, 2000）で、推奨される改訂版（Velicer, Eaton & Fava, 2000）なので既定にした。
偏分散が $10^{-10}$ 以下になったところで NaN として打ち切り、最小値は NaN を除いて探す。EFA.dimensions の 4 乗の
統計量は行列のべき乗である（記録するが従わない）。**EKC**: EFAtools と同じ Braeken & van Assen (2017)（残りの
分散の項に観測固有値を使う）。`PolychoricResult` を渡して `n_obs` を省くと、ペアごとの $n$ の最小値を使う
（欠測があるときに保守的）。
**照合（V25）**: `validation/r/m5_dimensionality_reference.R` は V24 の lavaan のポリコリック行列 6 つを EFAtools
1.1.0（`EKC`）、psych 2.6.9（`vss()$map`）、EFA.dimensions 0.1.9.3（`MAP`）、両方の MAP のべきの R による独立な
実装、PCA・SMC の固有値には `eigen()` に与える。選ばれる因子数も含めすべて $10^{-11}$ 以内で一致する。正規の
参照固有値（3 つの大きさ x PCA・SMC、各 20000 データセット）は EFAtools の `PARALLEL` と Monte Carlo 誤差の範囲で
一致する（固定のキーで平均は $|z|\le4.5$、95 パーセンタイルは $6\times10^{-3}$）。並べ替えの経路は性質（独立な
置換、2 因子の回復、キーによる再現性）で確かめる。

## 4. DETECT（`detect()`、`exploratory_detect()`。V27）

`_detect.py`（NumPy）に置く。flexpyirt §16.8 の計画の D4 にあたる。
**条件付き共分散**: Zhang & Stout (1999)。各ペアについて、総得点 $S$ が等しい群と残りの得点 $S-X_i-X_j$ が等しい
群のそれぞれで最尤の共分散を求め、群の割合で平均し、さらに 2 つを平均する（総得点による推定は負に、残りの得点に
よる推定は正に偏る）。群は `np.unique` と `np.bincount` でペアごとに 1 回で作る。生の項目得点を使い（多値項目も
可）、欠測のないデータを要する（そうでなければ得点が定義できない）。そのため `dimensionality()` は欠測があると
DETECT を使えないと示す。**指標**（sirt の `detect.index`）: DETECT $=100\,\overline{\delta\,\mathrm{ccov}}$、ASSI、
RATIO、MADCOV100、MCOV100 の、重みなしと $\sqrt{n_{ij}}$ で重み付けした版（欠測がなければ同じ。sirt の形に合わせて
残す）。sirt の既定の `conf.detect` の平滑化した（カーネルによる）条件付き共分散は実装しない。**探索的**: 行を半分に
分け（`rng` で `floor(N/2)` 行を推定用に引く。または指定）、推定用の半分の行列から作る
$d=\max(\mathrm{ccov})-\mathrm{ccov}$（最大値は sirt の `create.ccov` と同じく 0 の対角を含む）に Ward 法を当てる。
R の `hclust(method = "ward.D")` は Lance–Williams の Ward の更新を $d$ そのものに当て、SciPy の `"ward"` は 2 乗距離に
当てるので、SciPy には $\sqrt d$ を渡す（併合の高さは 2 乗すると R と同じ）。`cutree` はクラスタを最初に現れた順に
番号付けするので、SciPy の `fcluster` の出力も同じく付け直す。$k=2..$`max_clusters` の各分割を両方の半分で評価し、
`n_clusters` は推定用の半分での DETECT を最大にする（sirt と同じ選び方）。Zhang & Stout の遺伝的アルゴリズムによる
分割の探索は実装しない。**閾値**（docstring、ガイド）: DETECT $<0.2$ / $0.4$ / $1$、ASSI $0.25$、RATIO $0.36$
（Jang & Roussos, 2007; Zhang, 2007。flexpyirt §16.8 の 0.1 / 0.5 / 1 はここで訂正した）。**アンサンブル**:
`dimensionality()` は選んだ分割の検証用の半分での DETECT を問い (B) の行として加える（`detect_threshold` $=0.2$
未満なら支持）。`fit` がなくても使える。キーは 3 つ（平行分析、モデルに基づく平行分析、DETECT）に分ける。
**照合（V27）**: `validation/r/m5_detect_reference.R`（sirt 4.2.133）。シミュレーションの 2 次元（正しい分割と
誤った分割）、1 次元、Science（多値）のデータで `conf.detect(use_sum_score = TRUE, bias_corr = TRUE, smooth = FALSE)`、
3 クラスタのデータで `expl.detect(use_sum_score = TRUE, estsample = )`。条件付き共分散、5 つの指標（両方の
重み付け）、分割、両方の半分での指標は $10^{-12}$（実測 $10^{-14}$）で、Ward の併合の高さは $10^{-12}$ で一致する。
既定の `smooth = TRUE` のままだと、sirt の `conf.detect(use_sum_score = TRUE)` は総得点を標準化してから生の項目得点を
引いて残りの得点とするので、残りの得点の群が誤る（`sim_2d` で DETECT 1.60 が 6.73 になる）。精度のページに記録し、
従わない。

## 5. EFA と Hull 法（`efa()`、`hull()`。V28）

`_efa.py`（NumPy / SciPy）に置く。flexpyirt §16.8 の計画の D5 にあたる。
**推定**: 独自性 $\Psi$ から回転しない解を求め、SciPy の L-BFGS-B（`ftol=1e-15`、`gtol=1e-11`）で
$\psi_i=1-\mathrm{SMC}_i$ から $[0.005,\max(1,\mathrm{SMC})]$ の範囲（psych・EFAtools と同じ境界）で最小化する。
ULS: $F=\tfrac12\lVert R-\Psi-\Lambda\Lambda^\top\rVert_F^2$。$\Lambda$ は $R-\Psi$ の先頭の固有ベクトルを
$\sqrt{\max(\lambda,0)}$ でスケールしたもので、勾配は $\operatorname{diag}(\Lambda\Lambda^\top+\Psi-R)$（psych の
`FAgr.minres` で、この $F$ の厳密な勾配）。psych の minres、EFAtools の ULS と同じ（psych の `fit.residuals` は
非対角の残差だけを足すが、内点では最小点が同じ）。ML: $\Psi^{-1/2}R\Psi^{-1/2}$ の固有値による Jöreskog の
集約した不一致度で、勾配は $\operatorname{diag}(\Lambda\Lambda^\top+\Psi-R)/\psi^2$（psych の `FAfn` / `FAgr`）。
列は和が非負になるよう符号をそろえ、回転はしない。独自性が下限にあれば Heywood ケース（`efa()` は
`HeywoodWarning`）。許容誤差は R の `optim()` の既定より厳しいので、flexpyirt の目的関数値は EFAtools の解での値を
上回らない（テストで確認）。**適合度**（EFAtools の `.gof`）: CAF $=1-$（対角を 1 にした残差行列の KMO）。
$\chi^2$ は Bartlett の乗数 $N-1-(2p+5)/6-2k/3$ を使い、その $p$ 値も求める。CFI と RMSEA は補正しない
$F(N-1)$ と独立モデルの $-\log|R|(N-1)$（$df_0=p(p-1)/2$）から求める。$df=((p-k)^2-(p+k))/2$。
**Hull**（EFAtools の `HULL` / `.hull_calc`）: $k=0..J$ の行で、適合度は CAF、CFI、$1-$RMSEA のいずれか
（$k=0$ は $1-$KMO($R$)、0、$1-$RMSEA$_0$ と $df_0$）。適合度が増えない解を除き、両隣を結ぶ弦以下の内点が
なくなるまで除き、残った内点で scree 値 $st$ を求めて最大のものを選ぶ（凸包上の点が 3 未満なら、警告を出して
適合度が最大の解）。既定の $J$ は、正規データの SMC 固有値と平均の基準による平行分析（EFAtools の `efa_parallel`
の既定。ここでは `rng` で 1000 反復）の結果に 1 を足し、$df>0$ となる最大の $k$ で上を抑え、
$\min(3,\text{上限})$ 以上にする。`max_factors` を与えればその $J$ をそのまま使う。変数は 6 つ以上。表には解ごとの
`heywood` と `converged` を記録し、選ばれた解が不適切なら EFAtools と同じく `HeywoodWarning` を出す。
**アンサンブル**: 項目が 6 つ以上なら、`dimensionality()` は問い (A) に Hull（CAF、ULS、キーの 4 つ目の分割による
既定の $J$）を加える。
**照合（V28）**: `validation/r/m5_efa_hull_reference.R`（EFAtools 1.1.0、psych 2.6.9）。V24 のポリコリック行列のうち
変数が 6 つ以上のものと、シミュレーションの Pearson 行列 2 つ（3 因子、$N=500$。弱い 2 因子、$N=300$）。
$k\le J$ のすべてで ULS と ML について、含意する $\Lambda\Lambda^\top$ と共通性は $10^{-4}$（実測 $7\times10^{-5}$
以下で、最大は Heywood の解）、CAF / CFI / RMSEA は $2\times10^{-4}$、$\chi^2$ は相対 $10^{-4}$、$df$ は厳密に一致。
3 因子の行列での psych の `fa()` とも $10^{-4}$ で一致する。Hull は行列 x 推定法 x 適合度のすべてで EFAtools が
記録した $J$ を使い、選ばれる $k$ と凸包上の点は同じ、適合度は $10^{-5}$（実測 $1.4\times10^{-6}$ 以下）、$st$ は
相対 1%（小さな差の比なので。実測は絶対 $3.5\times10^{-3}$ 以下）で一致する。比較データ法は §6。

## 6. 比較データ法（`comparison_data()`。V29）

`_comparison_data.py`（NumPy）に置く。乱数は引数で渡すので、それを与えれば生成は決定的である（公開の関数が
`rng` から引く）。flexpyirt §16.8 の計画の D5b にあたる。$k=1..K$（$K$ は $df$ が正になる最大の $k$、または
`max_factors`）について、各変数の観測値を `n_population` 回復元抽出して並べ、標準正規の $F$ $(N,k)$ と
$E$ $(N,p)$ を引き、母集団を生成する（Ruscio & Kaczetow, 2008。EFAtools の `.simulate_cfm_empirical` と同じ）。
中間の $R^*$ を主因子法（EFAtools の `.paf_iter`。共通性の初期値は 1、固有値は絶対値、最大の変化に対する
基準 $10^{-3}$、`max_iter` 50）で分解し、負荷量を $[-1,1]$ に切り、最初の負荷量が負なら符号を反転して、得点
$F\Lambda^\top+E\,\mathrm{diag}(\sqrt{\max(1-h^2,0)})$ を作る。そのあと列ごとに順に、その列で行を並べ替え（安定な
整列）、その列を並べた復元抽出値で置き換える。$R^*$ は $R$ から始め、下三角の残差の RMSE が（1 から）改善する
限り残差 $R-R_{\text{rep}}$ だけ動かし、改善しなければ最良の $R^*$ からステップを半分にし、5 回続けて失敗したら
止める。母集団からデータと同じ大きさの標本を `n_samples` 個引き、Pearson 相関の固有値と観測固有値の RMSE を
記録する。$k$ 因子での RMSE が $k-1$ 因子より小さいと片側の Mann–Whitney 検定（連続性と同順位の補正を含む正規
近似で、$n\ge50$ での R の `wilcox.test` と同じ）が $\alpha=0.30$（Ruscio & Roche の推奨）で言う限り $k$ を
残し、最後に残した $k$ を結果とする。Pearson 相関のみ（Ruscio & Roche、EFAtools と同じ）、欠測のないデータ。
主因子法の負荷量は列の和が非負になるよう符号をそろえる。EFAtools は LAPACK の符号をそのまま使うが、効くのは
$k\ge2$ のときだけで、母集団の分布は変わらない。**アンサンブル**: 欠測がなければ `dimensionality()` は問い (A) に
CD を加える（キーの 5 つ目の分割）。**照合（V29）**: `validation/r/m5_comparison_data_reference.R`（EFAtools 1.1.0）。
$k=1..3$ の主因子法の負荷量は $\Lambda\Lambda^\top$ で $10^{-12}$、EFAtools 自身の乱数（同じシードで再現して
書き出す）による 1 因子の母集団は完全に同じ、EFAtools の RMSE の列に対する順位和の $p$ 値は相対 $10^{-10}$ で
一致する。3 因子、2 因子の順序、1 因子の 2 値のデータでの `CD()` とは、それぞれ 2 つのキーで同じ提案、平均 RMSE は
15% 以内（母集団の乱数が違うため）で一致する。

## 7. DIMTEST（`dimtest()`。V30）

`_dimtest.py`（NumPy）に置く。乱数は引数で渡し、公開の関数が `rng` から引く。flexpyirt §16.8 の計画の D6 に
あたる。
欠測のない 2 値データ、項目は 4 つ以上。**統計量**（Nandakumar & Stout, 1993）: 回答者を PT の得点で群に分け、
`min_group`（20）未満の群と $S_k\le0$ の群を除く。$T_L=K^{-1/2}\sum_k(\hat\sigma_k^2-\hat\sigma_{U,k}^2)/S_k$。
$\hat\sigma_k^2$ は AT の得点の（$J_k$ で割る）分散、$\hat\sigma_{U,k}^2=\sum_i\hat p_{ik}(1-\hat p_{ik})$、
$S_k^2=(\hat\mu_{4,k}-\hat\sigma_k^4+\hat\delta_{4,k})/J_k$、$\hat\delta_{4,k}=\sum_i\hat p_{ik}(1-\hat p_{ik})
(1-2\hat p_{ik})^2$。**バイアス補正**（Stout, Froelich & Gao, 2001）: 検定用の標本のカーネル平滑化した IRF（能力は
総得点の順位の $\Phi^{-1}(r/(N+1))$ で同順位は一様乱数で分け、$[-4,4]$ の 161 点上で帯域幅 $1.1N^{-1/5}$ の
ガウスカーネル。Ramsay, 1991）から、$\theta\sim N(0,1)$ と線形補間した IRF で検定用の標本と同じ大きさの 1 次元
データを $B$（100）個作り、$T=(T_L-\bar T_G)/\sqrt{1+1/B}$、$p=1-\Phi(T)$。**AT の選択**: `assessment_items` を
与えれば標本全体で検定する。与えなければ無作為な 3 分の 1（`training_fraction`）で AT を選び、残りで検定する
（Froelich & Habing, 2008 は別の部分標本を使う）。訓練用の条件付き共分散（§4）の Ward 法で 2〜6 群に分けた
ときのクラスタのうち、要素が 2 から `max_assessment_size`（項目の半分）までで、{AT, 残り} への分割の DETECT が
最大のもの。Froelich & Habing の条件付き共分散による AT の選択の考え方に沿うが、彼らの ATFIND そのものではなく、
そう明記する。**アンサンブル**: 欠測のない 2 値データなら `dimensionality()` は問い (B) に DIMTEST（$p\ge\alpha$）を
加える（キーの 6 つ目の分割）。そうでなければ使えない行とする。**照合（V30）**: 単独のプログラム DIMPACK 以外に
実装がないので、`validation/simulation/dimtest_study.py`（結果は `dimtest_study.json`）のシミュレーション研究で
受け入れる。2PL のデータ（傾き $U(0.8,1.8)$、切片 $N(0,0.8)$、項目を半分ずつ相関 $\rho$ の次元に置く）、各条件
200 反復、すべて既定値:

| データ | $N$ | $J$ | $\rho$ | $T$ の平均 | $T$ の SD | .05 で棄却 | .10 で棄却 |
|---|---|---|---|---|---|---|---|
| 1 次元 | 1000 | 20 | 1 | -0.07 | 0.72 | 0.005 | 0.020 |
| 1 次元 | 2000 | 20 | 1 | -0.12 | 0.71 | 0.010 | 0.030 |
| 1 次元 | 2000 | 30 | 1 | -0.03 | 0.73 | 0.010 | 0.025 |
| 2 次元 | 1000 | 20 | 0.7 | 4.85 | 0.97 | 0.990 | 0.995 |
| 2 次元 | 2000 | 20 | 0.7 | 6.94 | 0.88 | 1.000 | 1.000 |
| 2 次元 | 2000 | 30 | 0.8 | 6.50 | 0.86 | 1.000 | 1.000 |

（2026-11-25 に flexpyctt の NumPy の生成器で再実行した値。flexpyirt §16.18。移す前の JAX のキーでは、.05 での棄却率は
0.020、0.005、0.000 と 0.990、1.000、1.000 だった。）

検定は保守的である。$\bar T_G$ は $T_L$ の偏りを除く（$T$ の平均は 0 に近い）が、帰無仮説の下での $T_L$ の標準
偏差は 1 ではなく 0.72 程度なので、第 1 種の誤りは名目水準を下回る。相関 0.7〜0.8 の 2 次元に対する検出力は 0.99〜1。
したがって小さい $p$ は多次元性の強い証拠であり、大きい $p$ は一次元性の弱い証拠にとどまる（docstring とガイドに
書く）。$T_G$ のブートストラップ SD で標準化し直せば較正できるが、公表された統計量から離れるので行わない。各部分は
直接の計算と単体テストで比べ、`tests/test_dimtest.py` で研究の小さい版（帰無 20 反復、対立 10 反復）を実行する。

## 8. 観測得点による DIF（`mantel_haenszel()`、`mantel()`、`generalized_mantel_haenszel()`、`logistic_dif()`、`DIFResult`。V47）

IRT モデルを使わない DIF の検出。`flexpyctt.dif`（公開層: 入力の検査、マッチング得点、純化、表の組み立て）と `flexpyctt._dif`（検定する
1 項目についての NumPy のカーネル）に置く。API は flexpyirt の issue #102 で承認されたもので、変更点は最後に挙げる。

**共通の引数。** `data`、キーワード専用の `group`（行ごとに 1 つのラベル）、`focal`（ラベル 1 つ。一般化 MH では
リスト。それ以外の行はすべて参照群で、difR の `focal.name` と同じ）、`items=None`（検定する項目）、
`anchor_items=None`（合計をマッチング得点にする項目）、`matching=None`（外部のマッチング変数。`anchor_items`・
`purify` とは併用不可）、`purify=False`、`max_iter=10`、`p_adjust="none"`（`"holm"`、`"BH"`。有限の p 値に対する
R の `p.adjust`）、`alpha=0.05`。**マッチング**: アンカー項目の合計に、検定する項目がアンカーでなければそれを加える
（difR の規則）。異なる値ごとに 1 つの層（得点をまとめない）。検定する項目・アンカー項目の**欠測**は
`SpecificationError`（flexpyirt#102 のオーナー判断 3: 黙って行を除かずエラーにする。difR の `rowSums(na.rm = TRUE)` は
異なる項目の組の合計を混ぜてしまう）。

**純化**（difR の `purify = TRUE`、`puriadjType = "simple"`）: 第 0 段階は与えたアンカーですべての項目を検定する。
検出された項目（補正前の $p<\alpha$）がある間、次の段階はアンカーから検出された検定項目を除いたものを使う。検出
された組が直前と等しくなれば終わり（`converged=True`）、`max_iter` 段階で打ち切る（`ConvergenceWarning`。表は最後
の段階のもので、そのアンカーは 1 つ前の段階で検出された項目を除いたもの。difR と同じ）。第 0 段階で何も検出され
なければ `n_iter=0`、`converged=True`。利用者が `anchor_items` を与えたときは、その組からだけ検出された項目を
除く（difR は `anchor` を与えると `purify` を無視する。これは自然な拡張で、最終的なアンカーで直接呼んだ結果と一致
することをテストする）。最終的な `flagged` は `adj_p < alpha`。`flag_history` は difR の `difPur`（段階ごとに 1 行）。

**Mantel–Haenszel**（Mantel & Haenszel, 1959; Holland & Thayer, 1988）: $T_k>1$ の層を使う。
$\chi^2_{MH}=(\lvert\sum A-\sum E(A)\rvert-c)^2/\sum\operatorname{Var}(A)$ で、`continuity_correction=True` では
**常に** $c=1/2$（Holland & Thayer、difR。R の `mantelhaen.test()` は差が $1/2$ 未満なら $c=0$ にするので、DIF の
ない項目の統計量だけが変わる）。$\hat\alpha_{MH}=\sum AD/T/\sum BC/T$ で $A,B$ は参照群の正答・誤答数（したがって
$\hat\alpha_{MH}>1$ は参照群に有利）。$\ln\hat\alpha_{MH}$ の分散は Phillips & Holland (1987)。
$\Delta_{MH}=-2.35\ln\hat\alpha_{MH}$、$SE(\Delta)=2.35\,SE(\ln\hat\alpha)$。**ETS 分類**（オーナー判断 2: 既定は
本来の規則。Zieky, 1993; Zwick, 2012）: $\lvert\Delta\rvert<1$ または $\chi^2_{MH}$ が有意でない（`p_adjust` で
調整した p 値、つまり `flagged` と同じ値で $p\ge\alpha$。B と C の項目は必ず検出される。調整しなければ通常の $p$）なら A、
$\lvert\Delta\rvert\ge1.5$ かつ $(\lvert\Delta\rvert-1)/SE(\Delta)>z_{1-\alpha}$（$\lvert\Delta\rvert>1$ の片側検定。
$\alpha=0.05$ で 1.645。1.96 を使う文献もある）なら C、それ以外は B。
`classification="effect_size"`: A $<1\le$ B $<1.5\le$ C（difR が表示する分類は右閉区間で、ちょうど境界の値だけが
異なる）。$\hat\alpha_{MH}$ が 0 または無限大なら `FlexpycttWarning`（difR の "sparse data" の警告）。

順序のある項目の **Mantel**（1963）: 焦点群の得点の和をその超幾何分布の期待値と比べる。$df=1$、得点は項目の値。
0 / 1 項目では連続修正なしの $\chi^2_{MH}$ に等しい。**Liu–Agresti**（1996）の累積共通オッズ比
$\hat\psi=\sum_k\sum_{j<L}b_{jk}c_{jk}/T_k\big/\sum_k\sum_{j<L}a_{jk}d_{jk}/T_k$（$a,c$: カテゴリ $j$ 以下の参照群・
焦点群の人数）は **$\hat\alpha_{MH}$ と同じ向き**にする（0 / 1 項目では $\hat\alpha_{MH}$ になる。difR の
`liu_agresti_ccor()` はその逆数を返す）。標準誤差は **Liu & Agresti（1996）の式 (3)**。Robins, Breslow & Greenland
(1986) を一般化したもので、大きな層が一定数ある場合にも、小さな層が多数ある場合にも一致する（dually consistent）。
焦点群を彼らの第 1 行として
$\widehat{\operatorname{Var}}(\ln\hat\psi)=\sum_k\hat\xi_k\big/(\sum_k\sum_jb_{jk}c_{jk}/T_k)^2$、
$\hat\xi_k=\sum_j\hat\phi_{jjk}+2\sum_{j<s}\hat\phi_{jsk}$、
$\hat\phi_{jsk}=\frac{n_{Fk}n_{Rk}}{T_k^2}\{\hat\psi d_{sk}a_{jk}/n_{Fk}\,[1+(\hat\psi-1)a_{sk}/n_{Rk}]
+c_{jk}b_{sk}/n_{Rk}\,[\hat\psi-(\hat\psi-1)c_{sk}/n_{Fk}]\}$（$j\le s$）。合計得点でマッチングすると小さな層が
多数でき、層が大きいことを前提にしたデルタ法（多項分布の分散に推定値を代入した $\operatorname{Var}(S-\hat\psi R)/S^2$。
レビュー前の実装）はそこで大きく過小になる（1 層 4 人で約 40%）。この推定量はカテゴリの順序を逆にすると値が変わる。
Liu & Agresti は両方の順序の平均をとれるとし、実際の差はわずかだと述べている。flexpyctt は論文どおりカテゴリを
昇順に並べ、平均はとらない。difR 自身の標準誤差は誤っている（VALIDATION V47）。

**一般化 MH**（Landis, Heyman & Koch, 1978; Somes, 1986; Penfield, 2001）: 層（$T_k>1$）ごとの群 x 観測カテゴリの
$G\times C$ 表。カテゴリ $1..C-1$ の焦点群の度数を $T\,\mathbf p\otimes\mathbf q$ と比べ、共分散はそれらのセルに
制限した $T^2/(T-1)\,(D_p-pp^\top)\otimes(D_q-qq^\top)$。$df=(G-1)(C-1)$。合計した共分散行列が特異なとき（あるカテ
ゴリまたは群が情報のあるどの層にも現れない）は擬似逆行列を使い、$df$ をその階数にして `FlexpycttWarning` を出す。
複数の焦点群の 0 / 1 項目（difR の `difGMH()`）と名義カテゴリ（$I\times J\times K$ 表への `mantelhaen.test()`）の
両方に使う。

**ロジスティック回帰**（Swaminathan & Rogers, 1990; Zumbo, 1999）: モデル $S$、$S+G$、$S+G+SG$ を NumPy の
Newton–Raphson で推定する（依存を増やさない）。$\beta=0$ から、相対的なステップの最大値が $10^{-12}$ を下回るまで
（最大 100 回。分離などで情報行列が特異になると止め、`ConvergenceWarning` と擬似逆行列の共分散を返す）。尤度比
検定は逸脱度の差、Wald 検定は大きいほうのモデルで加えた係数について（difR の `criterion = "Wald"`）。Nagelkerke の
$R^2$ は $N$ = 行数、$\Delta R^2$ = 大きいモデル − 小さいモデル。Zumbo–Thomas（0.13、0.26）と Jodoin–Gierl
（0.035、0.07）の分類は $\Delta R^2$ だけによる（difR の表示と同じ。原典の規則は有意性も加えるが、それは `flagged`
列にある）。ETS の既定と違って有意性を組み込まないのは、これらの区切りが検定と並べて読む効果量として提案された
ためである。

**flexpyirt#102 で提案した API からの変更。** (1) `mantel()`: 標準誤差は対数のものなので、`cumulative_odds_ratio_se` を
`log_cumulative_odds_ratio_se` にし、新たに `log_cumulative_odds_ratio` を加えた（`mantel_haenszel()` の
`log_odds_ratio_se` と同じ）。(2) `mantel()`、`generalized_mantel_haenszel()`、`logistic_dif()` に `df` 列がある
（$df$ が固定の方法にもつけ、方法の違う表の列をそろえる）。(3) `DIFResult` に `flag_history`、`method` と
プロパティ `flagged_items` を加えた。(4) ロジスティック回帰の係数の列名は `group_coefficient`、
`group_coefficient_se`、`interaction_coefficient`、`interaction_coefficient_se`。(5) オーナー判断 4（標準化法、
STD P-DIF）: flexpyirt#102 の範囲の一覧どおり含めない。**この PR に含めないもの**: 順序ロジスティック回帰による DIF（flexpyirt#102 の
2 本目の PR）。

## 9. 古典的項目分析（V46）

`item_analysis.py`（NumPy / SciPy / pandas）に置く。flexpyirt からも再公開する（flexpyirt の issue #101）。API は issue で提案し、オーナーが承認したもの（2026-11-26）: `score_responses()`、
`item_analysis()`、`coefficient_alpha()`（`AlphaResult`）、`coefficient_omega()`（`OmegaResult`）、
`distractor_analysis()`。名前は flexpyirt のモデルに基づく `reliability()` と重ならない。**issue の案からの変更**
（シグネチャは変えていない）: `AlphaResult` に `confidence_level` を加え、`if_deleted` は `alpha` と
`standardized_alpha` の両方を持つ。`OmegaResult` に当てはめた相関行列 `correlation` と、ω の計算前に反転した `reversed_items` を加えた。`score_responses()`
はキーを列の順の列としても受け取る。
**欠測**（オーナーの決定事項 2。issue の既定 `"pairwise"`）: すべてペアワイズの共分散行列 $C$（R の
`cov(use = "pairwise")`）から計算し、標準化 α と ω にはペアワイズの相関行列（R の `cor(use = "pairwise")`。
`cov2cor(C)` ではない）を使う。psych の `alpha()` と同じ。`"listwise"` では先に欠測のある行を除く。どちらでも
1 つも回答のない行は除くので、`n_obs`（Feldt の $n$）は 1 つ以上回答した回答者の数になる。psych は `nrow(x)`。
**項目–合計・項目–残りの相関**は $c_j=\sum_l C_{jl}$ として $c_j/\sqrt{C_{jj}\,\mathbf 1^\top C\mathbf 1}$ と
$(c_j-C_{jj})/\sqrt{C_{jj}(\mathbf 1^\top C\mathbf 1-2c_j+C_{jj})}$。欠測がなければ $T$ と $T-x_j$ との Pearson 相関。
psych の `r.drop` は同じ式だが、`raw.r` は `rowSums(x, na.rm = TRUE)` との相関で、欠測を合計の 0 点として
扱うので α の元になる量ではない。flexpyctt は両方の列を $C$ から求め、`alpha`・`alpha_if_deleted` と同じ行列を
表すようにした。欠測時の `raw.r` との違いはテストし、文書に書いた。
**多分系列・双列相関**（`correlation="polyserial"`）: Olsson, Drasgow & Dorans (1982) の 2 段階推定量
$\hat\rho=r\,\sigma_j/\sum_k\phi(\tau_{jk})$。$\sigma_j$ は項目の SD（分母 $n$）、$\tau_{jk}$ は累積割合の
$\Phi^{-1}$ で、どちらも項目の観測された得点から求める。$r$ は上の Pearson の項目–合計（–残り）相関。polycor の
`polyserial(ML = FALSE)`、欠測のないときの psych の `polyserial()` と同じで、2 カテゴリでは
$r\sqrt{pq}/\phi(\Phi^{-1}(p))$。issue の「周辺から閾値、次に $\rho$」はこの推定量を指す。polycor の 2 段階版は
$\rho$ について最大化しない（最尤の多分系列相関 `ML = TRUE` は提供しない）。psych の `biserial()` は $p,q$ を
使いながら合計の分母 $n-1$ の SD で割るので、この値の $\sqrt{(n-1)/n}$ 倍になる（テスト済み）。$[-1,1]$ の外の
推定値は psych と同じく $\pm1$ にし、`HeywoodWarning` を出す。`difficulty` は平均を観測された最大得点で割るので、
得点は 0 以上とする。
**ω**: 1 因子の ULS 解（`efa()` のカーネル `_efa.fit_factors`）から、$s_j=\operatorname{sign}(\lambda_j)$（負荷量
0 なら $+1$）として $\omega_t=1-\sum_j(1-\lambda_j^2)/\mathbf s^\top R\mathbf s$。分母に観測された合計の分散を使う、
既定の `flip = TRUE` での psych の `omega.tot`（Revelle & Zinbarg, 2009）と同じ。負の負荷量の項目は psych と同じく
反転する（flexpyirt#127 のレビュー）。反転しないと、逆転項目が 1 つあるだけで $\mathbf 1^\top R\mathbf 1$ が小さくなり、誰も
使わない合計の ω になる。反転は黙って行わない。`FlexpycttWarning` で項目名を示し、`OmegaResult.reversed_items` に
挙げる。`loadings` と `correlation` は採点したままの項目のもの。先に項目を逆転採点しても同じ ω になる（テスト済み）。モデルが
含意する形 $(\sum\lambda)^2/((\sum\lambda)^2+\sum\psi)$ は完全に適合するときだけ一致し、返さない。`efa()` ではなく
カーネルを呼ぶので 3 項目でも計算できる（丁度識別のモデルで、`efa()` は自由度 0 として受け付けない）。Pearson の
行列が正定値でないとき（ペアワイズの行列ではありうる）は、`polychoric()` と同じく `smooth_correlation` で平滑化
する。issue に挙げた lavaan の 1 因子 CFA は参照に使わなかった。共分散行列への ML で、推定量も尺度も違う。
**キーによる採点と選択肢**: 選択肢はセルごとに `==` で比べるので、どの型のコードでもよい。誰も選ばない正答
選択肢は警告する（キーのコードや型の違いであることが多い）。`distractor_analysis()` では無回答は $T$ で 0 点。
各選択肢の統計量はその項目に答えた回答者で計算し、誰も選ばなくても正答は行に含める。得点群は R の
`cut(T, quantile(T, 0:G/G), include.lowest = TRUE)`（type 7 の分位点。同点は分けない）。分位点が重なったら、
黙って群を併合せずにエラーにする。
**照合（V46）**: `validation/r/ctt_item_analysis_reference.R`。§10 を参照。
**範囲外**（issue と同じ）: 階層 $\omega$、折半法、Guttman の $\lambda$、glb、上位下位 27% の識別指数、図（flexpyirt#103）、
ウェイト。

## 10. 検証

参照値は R で生成し、JSON として `tests/reference/` にコミットする。CI では R を実行しない。生成スクリプトは
`validation/r/` にあり（リポジトリのルートから実行する）、`validation/simulation/` には DIMTEST のシミュレーション
研究がある。ケースの番号は flexpyirt の一覧（V1〜V57）の続きで、同じケースは両方のリポジトリで同じ番号を持つ。

| ID | 内容 | 参照 | MS |
|---|---|---|---|
| V24 | ポリコリック・テトラコリック相関（シミュレーションのテトラコリック、2〜5 カテゴリの混在、10% MCAR、空のセルが 1 つの 2 x 2 表、LSAT7、Science）と平滑化 | lavaan の `lavCor(..., ordered = , missing = )`、psych の `polychoric(..., smooth = FALSE, correct = 0)`、`cor.smooth()` | M5 |
| V25 | 次元数: V24 の行列での EKC、MAP（2 乗と 4 乗）、PCA・SMC の固有値。正規データの平行分析の参照（3 つの大きさ） | EFAtools の `EKC()`、`PARALLEL()`、psych の `vss()$map`、EFA.dimensions の `MAP()`、4 乗版 MAP の R による独立な実装 | M5 |
| V27 | DETECT: 確認的（シミュレーションの 2 次元で正しい・誤った分割、1 次元、Science）と探索的（3 クラスタのシミュレーション、分割を指定） | sirt の `conf.detect(use_sum_score = TRUE, smooth = FALSE)`、`expl.detect(use_sum_score = TRUE, estsample = )` | M5 |
| V28 | EFA（ULS、ML。Hull の上限までのすべての $k$）と Hull 法（CAF、CFI、RMSEA）。V24 の行列のうち変数が 6 つ以上のものとシミュレーションの行列 2 つ | EFAtools の `EFA(method = , rotation = "none")`、`HULL(method = , gof = )`、psych の `fa(fm = "minres" / "ml", rotate = "none")` | M5 |
| V29 | 比較データ法: 主因子法、書き出した乱数による 1 因子の母集団の生成、順位和検定、3 因子の連続・2 因子の順序・1 因子の 2 値のデータでの `CD()` | EFAtools の `.paf_iter`、`.simulate_cfm_empirical`、`CD()`、`wilcox.test(x, y, "less")` | M5 |
| V30 | DIMTEST: 第 1 種の誤りと検出力（2PL。$N$ 1000〜2000、$J$ 20〜30。1 次元と相関 0.7〜0.8 の 2 次元。各 200 反復） | なし（DIMPACK 以外に実装がない）。`validation/simulation/dimtest_study.py` | M5 |
| V46 | 古典的項目分析（flexpyctt）: キーによる採点、α（ペアワイズとリストワイズ）、項目統計量、点双列・双列・多分系列相関、ω total（Pearson とポリコリック）、選択肢分析。シミュレーションによる 10 項目の多肢選択データ（欠測なし、5% MCAR）と 6 項目の 0〜4 のデータ（欠測なし、6% MCAR） | なし（mirt ではない）: psych `alpha()`、`score.multiple.choice()`、`responseFrequency()`、`polyserial()`、`biserial()`、`fa()`、`omega(nfactors = 1)`。ltm `cronbach.alpha()`、`biserial.cor()`。polycor `polyserial(ML = FALSE)`。lavaan `lavCor()`。選択肢の統計量は素の R | flexpyctt |
| V47 | 観測得点による DIF（flexpyctt、flexpyirt#102）: 連続修正の有無、Holm / BH、純化、利用者のアンカー、外部のマッチング変数を含む Mantel–Haenszel（`difR::verbal`、均一・不均一 DIF を入れたシミュレーションの 2PL）。3 群と名義カテゴリの一般化 MH（シミュレーションの段階反応）。Mantel 検定と Liu–Agresti のオッズ比。ロジスティック回帰（both / uniform / non-uniform、尤度比 / Wald、純化、アンカー、マッチング） | difR 6.1.0 の `difMH()`、`difGMH()`、`difLogistic()`、`difMantel.poly()`、`liu_agresti_ccor()`。`stats::mantelhaen.test()`。R での直接の実装: `validation/r/ctt_dif_reference.R`。Liu & Agresti（1996、表 1）の計算例 | — |

各ケースについての記述:

- **ポリコリック相関（V24、M5）**: mirt との照合ではない。順序変数に対する lavaan の `lavCor`（`nlminb` に解析的な
  勾配とヘッセ行列を与えた two-step 最尤法、$|\rho|\le0.999$、既定の `zero.add = c(0.5, 0)`）を厳密な参照値とする。
  lavaan が非収束の警告を出したらスクリプトを止める。psych の `polychoric` は既定の許容誤差の `optimize()` と
  `mnormt::sadmvn` を使うので、1e-3 までの照合とし、欠測のないデータで、空のセルを補正したペアを除いて比べる
  （`global = TRUE` の psych は補正しない）。カテゴリ数が異なると psych はペアごとの閾値を使い、欠測があると結果が
  ずれる（MCAR のケースで 6e-3）ため、そのケースは lavaan とだけ比べる。さらに各ペアの two-step の勾配が推定値で
  0 になることも確かめる。2 変量正規分布の分布関数は Plackett の積分と単体テストで比べる。
- **次元数（V25、M5）**: 決定的な量は V24 の lavaan の行列から R で計算するので、両側が同じ入力を見る。
  EFA.dimensions と EFAtools を両方読み込むと EFA.dimensions の `PARALLEL()` が EFAtools のものを隠すため、
  スクリプトでは `EFAtools::` を明示する。EFA.dimensions の 4 乗版 MAP は $(\operatorname{tr}(R^{*4})-p)/(p(p-1))$
  （行列のべき乗。このデータで 1 を超える）なので、4 乗版は要素ごとの独立な実装と比べ、EFA.dimensions の値は記録
  するだけにする。正規データの参照は両側とも確率的なので、テストは固定のキーを使い、平均は $z$（反復から求めた標準
  誤差、独立な 2 つの平均なので $\sqrt2$ 倍）で、95 パーセンタイルは絶対誤差で比べる。
- **DETECT（V27、M5）**: sirt の `conf.detect()` は `smooth = FALSE` で呼ぶ。既定の `smooth = TRUE` で
  `use_sum_score = TRUE` だと、標準化した総得点から生の項目得点を引くので、残りの得点の群が誤る（DESIGN §4）。
  `expl.detect()` は自分で `smooth = FALSE` にする。その戻り値の `detect.unweighted` は検証用の半分の指標だけを
  持つので、推定用の半分の指標はスクリプトで推定用の行に `conf.detect()` を当てて計算し直す。
- **EFA と Hull（V28、M5）**: 回転しない負荷量は、含意する $\Lambda\Lambda^\top$（列の符号と等しい固有値の順に
  よらない）で比べる。`HULL()` の上限 $J$ は乱数を使う平行分析から決まるので、スクリプトは `settings$n_fac_max` を
  記録し、テストはそれを `max_factors` として渡す。EFAtools の `optim()` による解は flexpyirt より収束が粗いので、
  テストでは EFAtools の $\psi=1-h^2$ での目的関数値を flexpyirt の値が上回らないことも確かめる。残る差は Heywood の解
  （独自性が境界にある）で最も大きく、Hull は真の因子数を超える $k$ でそれを含む。
- **比較データ法（V29、M5）**: CD は両側とも乱数を使う。生成器は、EFAtools の乱数を R で再現し（`set.seed` の
  あと、生成器と同じ順に `sample()` / `rnorm()` を呼ぶ）、書き出して flexpyirt に与えることで厳密に確かめる。負荷量の
  符号が生成器で決まる 1 因子では厳密に一致する。選択は答えがはっきりしたデータで確かめ、平均 RMSE は各側が別の
  母集団を引くので緩く比べる（採用された因子数では 15%。棄却された追加の因子では 60%。こちらは 1 つの母集団による
  ばらつきがずっと大きく、EFAtools の値 0.048 に対して 6 つのシードで 0.044〜0.072 になる）。JSON の `auto_unbox` は要素が 1 つの $p$ 値のベクトルをスカラーにするので、スクリプトは
  `as.list()` で書く。
- **DIMTEST（V30、M5）**: 参照値との照合ではなくシミュレーション研究で受け入れる（表は DESIGN §7）。200 反復で
  .05 での第 1 種の誤りは 0.5〜1.0%（保守的。ブートストラップは偏りを除くが、帰無仮説の下での $T$ の SD は 0.72 程度）、
  検出力は 0.99〜1.00。`uv run python validation/simulation/dimtest_study.py`（約 4 分）で再実行できる。テストは帰無
  20 反復と対立 10 反復を実行する。
- **古典的項目分析（V46、flexpyctt、flexpyirt#101）**:
  `validation/r/ctt_item_analysis_reference.R`（R 4.6.2、psych 2.6.9、ltm 1.2.0、
  polycor 0.8.2、lavaan 0.7.3）が `tests/reference/v46_item_analysis.json` を書き、
  `tests/test_item_analysis.py` で照合する。シミュレーションの 4 例: 多肢選択 10 項目（$N=500$、
  正答と誤答の引きつけ方を変え、1 項目は 5 選択肢）の欠測なしと 5% MCAR、0〜4 で採点する 6 項目（$N=300$、
  段階反応モデル）の欠測なしと 6% MCAR。ほかに丁度識別の ω を見る 3 項目と、0〜4 の欠測なしデータの 1 項目を逆転採点したもの（psych の既定の
  `flip = TRUE` がその項目を反転する。`flip = FALSE` の値も記録し、0.18 低く、反転しない式と一致する）。観測された差: 閉じた式（α、標準化 α、
  項目を除いたときの値、Feldt の上下限、平均、SD、項目–合計・項目–残りの相関、点双列・多分系列・双列相関、
  選択肢の割合・点双列相関・平均得点・群ごとの割合）は $6\times10^{-15}$ 以下、キーによる得点は完全一致、
  ω total は $7\times10^{-8}$ 以下、負荷量は $1.4\times10^{-6}$ 以下（psych の `fa()` が minres の最適化を早く
  止めるため）、lavaan とのポリコリック行列は $2\times10^{-8}$ 以下。反転する項目は psych が反転するものと一致し、反転したときの ω は
  手で逆転採点したデータの ω と一致する。既知の違い（それぞれテストで確かめる）:
  (1) psych の `raw.r` は `rowSums(na.rm = TRUE)` との相関なので、欠測があるときは R で求めた $C$ に基づく
  `colSums(C)/sqrt(diag(C) * sum(C))` と比べ、`raw.r` そのものとは欠測なし・リストワイズのときだけ比べる
  （DESIGN §9）。`r.drop` は psych でも $C$ に基づくので全例で比べる。(2) ltm の素の
  `cronbach.alpha(na.rm = TRUE)` は合計の分散を欠測のない行から、項目の分散をすべての観測された得点から求め、
  どちらの規則でもない。欠測のないデータでだけ比べる（標準化 α は欠測のない行を使い、`"listwise"` と一致する）。
  (3) psych の `biserial()` は 2 段階の双列相関の $\sqrt{(n-1)/n}$ 倍になる。(4) psych の `polyserial()` は欠測の
  ある行も $n$ に数えるので、欠測のないデータで比べる。欠測があるときは 2 段階の係数 $\sigma_j/\sum_k\phi(\tau_{jk})$
  を R で計算し直して比べる。(5) 選択肢の統計量にはインストール済みの R 実装がない（TAM の `tam.ctt3` は未導入）。
  スクリプトが定義どおり素の R で計算し、選択肢の割合は psych の `responseFrequency()` とも一致する。issue に挙げた
  lavaan の 1 因子 CFA は使わなかった（共分散行列への ML は推定量も尺度も違う）。
- **観測得点による DIF（V47、flexpyctt、flexpyirt#102）**: 参照値は difR 6.1.0 と base R によるもので、
  `tests/reference/v47_dif.json` にコミットする。一致の程度: MH の $\chi^2$、$\hat\alpha_{MH}$ と
  Phillips–Holland の分散、一般化 MH の統計量（difGMH と、群 x カテゴリ x 層の表への `mantelhaen.test()`）、Mantel の
  統計量とオッズ比は相対 $10^{-10}$ まで（どれも閉じた式）。純化の経過（`difPur`）、反復回数、収束の判定、最終的な
  検出結果は一致。ロジスティック回帰: `difLogistic()` は既定の `epsilon = 1e-8` で `glm()` を呼ぶので、このデータでは
  係数と統計量が最尤推定値から相対 $2\times10^{-4}$ までずれる。そこでスクリプトは `stats::glm.control` を
  `epsilon = 1e-14` のものに置き換える。すると尤度比統計量、$\Delta R^2$、係数は $10^{-8}$ まで、標準誤差と Wald
  統計量は $10^{-6}$ まで一致する（`glm()` の `vcov()` は 1 つ前の反復の IRLS の重みを使い、逸脱度の基準は
  $\sqrt{\epsilon}$ 程度のステップで止まる）。テストで確認している既知の差: (1) `mantelhaen.test()` は
  $\lvert\sum A-\sum E(A)\rvert<1/2$ のとき連続修正を省く（difR と flexpyctt は省かない。シミュレーションのデータで
  3 項目）。(2) `difMantel.poly()` は出力を丸め（統計量は小数 3 桁、オッズ比は 4 桁）、純化では検出された項目を
  その項目自身のマッチング得点からも外す。そこで Mantel の統計量と純化の経過は、difR のほかの方法と同じ規則で R で
  計算し直し、`difMantel.poly()` は丸めた値での照合にとどめる。(3) `liu_agresti_ccor(se = "liu_agresti")` は
  $T=\sum ad/N-\hat\psi\sum bc/N$ の焦点群の累積度数による微分を $a/N-\hat\psi b/N$ としており（$a/N$ の符号が誤り）、
  $\sum ad/N$ ではなく $\sum bc/N$ で割るので、その `SE_log_Psi` は $SE(\ln\hat\psi)$ でも $SE(\hat\psi)$ でもない
  （$\psi=1.48$、3 層、$N=20000$ を 300 反復したモンテカルロの SD 0.0296 に対して 0.031。`verbal` のある項目で
  Phillips–Holland の 0.36 に対して 0.14）。flexpyctt と R スクリプトの `la_eq3()` は代わりに Liu & Agresti（1996）の
  式 (3) を使う。小さな層が多数ある場合にも妥当な推定量である（DESIGN §8）。R の実装は論文の記法で Python とは
  独立に書いたもので、論文の計算例（表 1、28 施設: $\ln\hat\theta=-1.153$、SE .571）を再現する。Python のテストも
  Mantel の 4.84 を含めて同じ公表値を確認する。推定値を代入するデルタ法（レビュー前の実装。層が大きい確認では
  モンテカルロの 0.0296 に対して 0.0291）はこの例で 0.375 になり、1 層 4 人では約 40% 過小になる。Python の
  モンテカルロのテスト（4 人と 10 人の層が 150 個、4 カテゴリの比例オッズの項目、400 反復）で式 (3) の SE の平均が
  $\ln\hat\psi$ のモンテカルロ SD と 10% 以内で一致すること、別のテストで層が大きい 0 / 1 項目では
  Phillips–Holland の標準誤差と 1% 以内で一致することを確認する。
  (4) `liu_agresti_ccor()` は flexpyctt の $1/\hat\psi$ を返す（difR の向き: 1 より大きいと焦点群に有利）。
