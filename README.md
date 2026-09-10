# Flexible EEG Feature Extraction Pipeline

A configurable Python pipeline for EEG feature extraction, developed from Pedro
Benedetti's processing script. The project separates acquisition-specific
preprocessing, experimental event definitions, feature estimators and exports.
It is designed for an editable Python configuration and a regular VS Code workflow.

This implementation grew out of a comparison with a MATLAB methodology adapted
from another experimental protocol. MATLAB is a methodological reference; exact
numerical agreement is not assumed when estimators or analysis intervals differ.

## Current scope

- One shared preprocessing and ICA solution per recording.
- Configurable conditions, event codes, block durations and analysis windows.
- A single region definition shared by all estimators.
- Selectable spectral, wPLI, wSMI, permutation entropy, Lempel–Ziv complexity and
  transfer entropy features.
- Subject-level tables, detailed arrays, interval records and quality information.
- Synthetic regression tests and an end-to-end integration test.

Reshaping, normalization, PCA and statistical modelling are outside this package.
The existing downstream scripts still need a separate compatibility review.

## Project structure

| File | Responsibility |
|---|---|
| `ejecutar_pipeline.py` | Main orchestration, shared cleaning, cache and feature selection |
| `config_protocolo.py` | Explicit paths, selected subjects/features, bands, events and estimator settings |
| `config_sujetos.py` | Recording-specific bad channels and ICA component exclusions |
| `config_regiones.py` | Shared channel-to-region definitions |
| `segmentacion.py` | Event decoding, block validation and sample intervals |
| `features.py` | Feature estimators and regional aggregation |
| `salidas.py` | Atomic workbook exports, NPZ arrays and JSON metadata |
| `verificar_pipeline.py` | Synthetic segmentation, numerical and export tests |
| `verificar_integracion.py` | Full synthetic FIF workflow and ICA-cache test |
| `referencia/featureExtraction_original.py` | Unmodified source used for regression comparisons |

Configuration remains explicit and editable. `DATA_DIR`, `OUTPUT_DIR` and
`CACHE_DIR` are absolute paths; recording names are selected through
`SUBJECTS_TO_RUN` and configured in `config_sujetos.py`. These paths must match
the local computer. The entry point is `ejecutar_pipeline.py`, using the selected
Python interpreter in VS Code. No command-line arguments are required.

The initial selection is the first recording and `FEATURES_TO_RUN = ["wpli"]`.
The complete feature list is preserved in a commented configuration line.

## Experimental intervals

The current protocol has three conditions:

| Code | Condition | Block length | Intermediate marker | Windows |
|---|---|---|---|---|
| 40 | Resting | 60 s | Code 40 at 30 s | 12 × 5 s |
| 60 | REY | 60 s | Code 60 at 30 s | 12 × 5 s |
| 100 | AUT | 60 s | Code 100 at 30 s | 12 × 5 s |

The second occurrence of a condition code verifies the midpoint of the same
block. It does not create a second independent experimental condition. Windows
use half-open sample intervals, `[start, stop)`, without duplicating boundaries.
The original Status channel is retained; artificial window markers are not needed.

Three interval definitions are supported:

- `fixed`: onset code, fixed duration and optional intermediate markers.
- `until_marker`: separate onset/offset codes and optional duration limits.
- `event`: an interval relative to an event, defined by `tmin_s` and `tmax_s`.
  Its entire interval is an individual trial.

For fixed and marker-delimited blocks, individual-feature windows are generated
using `WINDOW_SECONDS`. Overlapping blocks are rejected. Missing markers,
unexpected durations or midpoint deviations invalidate the affected condition
and are recorded. Other valid conditions can continue.

The current midpoint tolerance is 0.1 s. Tolerance validates timing; it does not
move the windows. The default remainder policy rejects incomplete windows.
An optional `drop` policy records discarded final samples for individual-window
features. wPLI independently requires complete subepochs and at least the
configured minimum count.

## Shared preprocessing and ICA

The original preprocessing order is preserved: 50 Hz notch filtering,
EXG1/EXG2 reference, optional bad-channel interpolation, 1–30 Hz FIR filtering
and resampling to 500 Hz. The current implementation assumes BioSemi128 channel
names and a Status channel. Adapting event rules does not automatically adapt
hardware, montage, reference or filter choices to another acquisition system.

ICA is fitted to EEG channels once per subject. Manually specified component
exclusions are applied once, and all feature estimators use that cleaned signal.
Different marker representations do not require separate ICA fits when the EEG
samples supplied to ICA are identical.

The cleaned FIF and ICA solution are cached. Reuse requires matching input hash,
preprocessing settings, exclusions, relevant software versions and cleaning code.
The cache files are checked against their recorded hashes. Each analysis run
still has its own output directory and configuration snapshot.

## Feature definitions

Default bands are delta 1–4 Hz, theta 4–8 Hz, alpha 8–12 Hz and beta 12–30 Hz.
The original inclusive band-boundary conventions are retained.

| Feature | Computation unit | Implementation retained |
|---|---|---|
| Spectral parameterization | Each 5 s window | MNE PSD and specparam; channel results averaged within regions |
| wPLI | Entire 60 s block, using twelve 5 s epochs jointly | MNE-Connectivity, Fourier estimation and frequency-band averaging |
| wSMI | Each 5 s window | Original symbolic estimator, with channel-pair and regional results |
| Permutation entropy | Each 5 s window | Original symbolic entropy estimator |
| Lempel–Ziv complexity | Each 5 s window | Original LZ78-based computation |
| Transfer entropy | Each 5 s window | Original Gaussian-copula conditional mutual information computation over lags |

LZC and TE remain broadband measures in this version. Their values are repeated
in band-labelled aggregate rows for compatibility with the previous table layout;
those repeated values are not independent band-specific measurements. Individual
estimators retain their original additional filters and parameters.

### wPLI

The main estimator is `mne_connectivity.spectral_connectivity_epochs` with
`method="wpli"`, `mode="fourier"` and `faverage=True`. The original per-band IIR
filtering is retained. One block produces one connectivity estimate per band,
computed across all twelve epochs; it does not produce twelve independent wPLI
observations. If a different protocol has several blocks, each block is estimated
separately and the aggregate table averages those block results.

For the all-pairs MNE output, the calculated lower triangle is mirrored once.
A calculated entry is never averaged with an uncomputed zero in the opposite
triangle. Channel self-connections are zero. Regional diagonal entries summarize
distinct channel pairs within a region; a one-channel region therefore has no
intra-regional estimate. The aggregate table retains the ten unique interregional
pairs for the current five-region configuration. Full matrices remain in details.

This is conventional wPLI, not `wpli2_debiased`. Finite sample size matters, and
more epochs do not by themselves establish reliability. Consecutive windows are
not additional independent subjects. The MNE method computes connectivity across
epochs under assumptions about the spectral process and its consistency across
epochs. See the [MNE-Connectivity documentation](https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html)
and [Vinck et al. (2011)](https://pubmed.ncbi.nlm.nih.gov/21276857/).

### Spectral quantity: `spec_period_*`

The exported name changes from `spec_theta_*` to `spec_period_*`, because the
same calculation is used for all selected bands. Its numerical definition is
preserved:

$$
Q_B = \int_B 10^{\log_{10} P(f)-A_{\log_{10}}(f)}\,df
    = \int_B \frac{P(f)}{A_{\mathrm{linear}}(f)}\,df.
$$

Here, `P` is the observed PSD and `A` is the fitted aperiodic background.
For the fixed aperiodic model, the log-background is
`offset - exponent * log10(f)`.

The measure integrates the PSD-to-background ratio, including residual spectral
deviations even when no Gaussian peak is detected. It is not absolute periodic
power in µV² and not the area of the fitted peaks. The ratio is dimensionless;
its frequency integral has units of Hz. A spectrum equal to its background gives
approximately the integrated band width, rather than zero. Values across bands
of different widths are consequently not directly equivalent.

Internal `theta_power_*` array names are retained for regression compatibility.
They refer to the requested band, not necessarily theta. No change to this
scientific definition is introduced by the refactor.

## Outputs and audit trail

Each execution creates a separate directory containing:

- `EEG_features_subject_level.xlsx`: aggregate feature table in `subject_level`,
  with `subject`, `band` and `condition` identifiers. Supporting sheets contain
  traceability, intervals, issues, quality summaries and region membership.
- `detalles/`: pickle-free NPZ arrays and JSON descriptors with channel/region
  names, array axes, interval definitions, parameters and versions.
- `eventos/`: detected events and the validated block/window boundaries.
- `configuracion.json`, `codigo/` and `ejecucion.log`: effective configuration,
  a source snapshot and execution log.
- Optional CSV export from the same aggregate table, with comma delimiters,
  decimal points and explicit floating-point formatting.

Quality tables report finite, NaN and infinite values, plus the number of valid
window/block contributions to each regional aggregate. Missing values are not
replaced by zero. Failed calculations and partial results are recorded.

Workbook writes use a temporary file followed by replacement. An export failure,
such as an Excel file locked on Windows, stops the run rather than silently
replacing prior results with an empty table.

MNE may warn that an `EpochsArray` has no annotations to add to metadata. The
pipeline records this warning because its interval metadata are stored separately.
This can produce `completed_with_issues` even when all calculations completed.
Inspect the issue messages and quality tables, not only the final status string.

## Verification status

The included synthetic tests passed for:

- Current and alternative interval definitions, invalid markers and remainders.
- Joint twelve-epoch wPLI, compared with explicitly requested MNE channel pairs.
- Spectral, wSMI, PE, LZC and TE outputs against the original implementation on
  identical windows, with `1e-12` numerical tolerance and matching NaN positions.
- Pickle-free array exports, identifier preservation and atomic workbook writes.
- A complete synthetic FIF run and reuse of cached cleaning without fitting ICA again.

The synthetic regression uses two windows, with TE lags up to 10 ms to keep the
test small. The separate wPLI check uses twelve 5 s epochs. These tests do not
establish physiological validity or statistical reliability.

A first real-data run on the user's Windows/Python 3.10 system completed all
three conditions and four wPLI bands. The submitted workbook recorded complete
60 s blocks, twelve epochs per estimate, no missing regional channels and no
non-finite connectivity values. This is an initial execution check, not a full
validation across subjects or a verification of the other features on that system.

The synthetic test environment used Python 3.12.14, MNE 1.6.1, MNE-Connectivity
0.6.0, NumPy 1.26.4, SciPy 1.13.1, specparam 2.0.0rc3, pandas 2.2.3 and openpyxl
3.1.5. Python 3.10 syntax was checked separately. Use the existing analysis
environment and record its versions; this list is not a request to upgrade.

Before interpreting condition effects, assess estimator variability, matched
sample counts, possible residual artifacts and sensitivity to interpolated
channels. Suitable future checks include null-surrogate comparisons and
within-recording stability analyses. No inferential conclusion follows from
values merely falling inside the valid numerical range.

## Attribution

Original processing implementation: Pedro Benedetti (pbenedetti@itba.edu.ar). The reference file is kept
for traceability. Cite the methods and libraries actually used in an analysis;
this refactoring is not a new connectivity or spectral estimator. No new software
license is assigned by this documentation update.
