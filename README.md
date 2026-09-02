# PolyFormer

Research code, public data, and reference results accompanying the manuscript
**“Learning efficient representations of complex constraints for scalable optimization.”**

PolyFormer is a physics-informed machine-learning framework for replacing complex
feasible-region constraints with compact polytopes. For a parameterized feasible
region

```math
\Omega(\boldsymbol\theta)
=
\{\mathbf x\mid \exists \mathbf y:
g(\mathbf x,\mathbf y;\boldsymbol\theta)\leq 0\},
```

PolyFormer learns

```math
\mathcal P(\boldsymbol\theta)
=
\{\mathbf x\mid
\mathbf A(\boldsymbol\theta)\mathbf x
\leq
\mathbf b(\boldsymbol\theta)\}.
```

The learned inequalities can be inserted directly into downstream optimization
models. This reduces the number of variables and constraints while retaining a
controlled balance between:

- **feasibility error**, which measures outward overestimation and the risk of
  admitting decisions outside the original feasible region; and
- **optimality error**, which measures inward undercoverage and the loss of
  potentially useful decisions.

The repository implements three representations:

- `PreTrainNet`: a fixed polytope for a fixed feasible region;
- `BiasNet`: parameterized offsets with fixed facet directions; and
- `FullNet`: parameterized facet directions and offsets.

The paper evaluates PolyFormer on resource aggregation, transmission–distribution
(T–D) network optimization, and distributionally robust chance-constrained
(DRCC) portfolio optimization. Supplementary Note 2 also studies polygon,
ellipse, nonconvex, hypercube, and ball benchmarks.

## Main reported results

| Application | Original formulation | PolyFormer formulation | Main reported outcome |
|---|---|---|---|
| 1,000-resource aggregation | 57,696 constraints | 96 inequalities over 24 aggregate variables | 99.83% constraint removal; final mean feasibility error `7.1e-15` |
| 105-resource mixed aggregation | 5,184 continuous variables, 336 binary variables, and 6,291 constraints | 24 continuous variables, no binary variables, and 96 inequalities | 99.54% continuous-variable reduction; 98.47% constraint reduction; final mean feasibility error `1.1e-5` |
| Largest T–D case | 715,055 constraints and 477,691 variables; 1,476 s and 821 MB online solve | 2,239 constraints and 1,785 variables; 0.23 s and 3.50 MB | More than 6,400-fold speedup and 99.6% memory reduction |
| Largest DRCC case | 1,044,497 constraints and 1,034,656 variables | 1,617 constraints and 400 variables | About 708-fold speedup and 99.87% memory reduction |

## Paper-to-repository map

| Paper component | Experiment | Code | Input data | Reference results |
|---|---|---|---|---|
| Fig. 1, Methods, Supplementary Note 1 | PolyFormer formulation, errors, networks, and training | `Simulator/Approximator.py`, `Simulator/Counter.py`, `Simulator/Plotter.py` | Generated in memory or supplied by each case | Case-specific directories under `results` |
| Extended Data figures, Supplementary Note 2 | Polygon, ellipse, nonconvex region, hypercube, and ball | `Simulator/cases/basic_cases.py` and `Simulator/runners/main_{polygon,ellipse,nonconvex,cube,ball}.py` | Generated in memory | `results/polygon`, `results/ellipse`, `results/nonconvex`, `results/cube`, `results/ball` |
| Fig. 2, Supplementary Note 3 | Aggregation of EVs, HPs, and BSSs | `Simulator/cases/aggregation_case.py`, `Simulator/runners/main_agg.py` | `data/aggregator_data`, `data/profiles_data` | `results/aggregation` |
| Fig. 3, Supplementary Note 4 | Distribution-network projection and T–D optimization | `Simulator/cases/TD_case.py`, `Simulator/cases/DS_case_3phase.py`, `Simulator/runners/main_ds.py`, `Simulator/runners/main_ds_3phase.py` | `data/TD_OPF` and `data/real_dis_data` | `results/ds_proj`, `results/ds_proj_original` |
| Fig. 4, Supplementary Note 5 | DRCC portfolio optimization | `Simulator/cases/DRCC_case.py`, `Simulator/runners/main_drcc.py` | `data/DRCC` | `results/DRCC` |

`Simulator/testers` contains evaluation programs used to produce or inspect paper
results; it is not a `pytest` suite. `Simulator/drawers` contains plotting scripts
used for manuscript figures. The documented runners under `Simulator/runners`
are the recommended entry points for training and smoke checks.

## Repository layout

```text
PolyFormer/
├── data/
│   ├── aggregator_data/       # Building records used to parameterize HPs
│   ├── profiles_data/         # Load, PV, and temperature profiles
│   ├── TD_OPF/                # Balanced DS and transmission-system cases
│   ├── real_dis_data/         # De-identified three-phase measurements
│   └── DRCC/                  # Training and out-of-sample return samples
├── results/                   # Reference results and default runner output root
└── Simulator/
    ├── cases/                 # Original feasible-region models
    ├── runners/               # Documented command-line entry points
    ├── testers/               # Paper evaluation programs
    ├── drawers/               # Figure-generation scripts
    ├── Approximator.py        # Networks, loss construction, and training
    ├── Counter.py             # Error and sensitivity calculations
    ├── Plotter.py             # Shared visualization utilities
    └── validate_release.py    # Data/result integrity checks
```

## Installation

### Python environment

Python **3.12** is required. The source uses Python 3.12 f-string syntax.

The tested dependency ranges are recorded in `requirements.txt`. A Conda
environment is supplied because the T–D workflows require an IPOPT executable:

```bash
conda env create -f environment.yml
conda activate polyformer
```

For an existing Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
```

NumPy is constrained to `<2.0` to match the tested scientific Python stack.
CUDA is optional; smoke checks can run with PyTorch on CPU.

### Optimization solvers

Solvers are external to the Python package installation.

| Solver | Used by | Requirement |
|---|---|---|
| **Gurobi** | Aggregation, DRCC, and auxiliary polytope problems | A working Gurobi installation and license |
| **IPOPT** | Original nonlinear distribution-network models and nonlinear geometry cases | Installed by the supplied Conda environment |
| **CPLEX** | Canonical polygon and ellipse cases | A working CPLEX installation and license |

Check solver visibility after activating the environment:

```bash
python -c "from pyomo.environ import SolverFactory; print({s: SolverFactory(s).available(False) for s in ('gurobi','ipopt','cplex')})"
```

The core application checks require Gurobi and IPOPT. Solver failures and
non-optimal termination conditions are reported as errors rather than converted
to zero error values.

## Validate the checkout

Run all commands from the repository root.

### 1. Validate data and reference results

```bash
python -m Simulator.validate_release
```

The validator checks release-critical files and schemas, including:

- aggregation MAT, NPZ, and CSV inputs;
- the eight balanced distribution cases and three transmission cases;
- both required English three-phase workbooks in `data/real_dis_data`;
- DRCC training/test CSV shapes, group labels, means, and finite values;
- the two complete 27-case T–D result workbooks and their required sheets; and
- the expected group weights and test results for the four main DRCC cases.

The three-phase workbooks are required public inputs. Their absence or a schema
mismatch is a validation failure.

### 2. Run one update for the three main applications

```bash
python -m Simulator.runners.smoke_test --case all
```

Individual application checks are also available:

```bash
python -m Simulator.runners.smoke_test --case aggregation
python -m Simulator.runners.smoke_test --case td
python -m Simulator.runners.smoke_test --case drcc
```

| Smoke case | Work performed |
|---|---|
| Aggregation | Tiny mixed fleet with 2 EVs, 1 HP, and 1 BSS; one PreTrainNet update |
| Balanced T–D | `case10ba_ds`; one voltage-parameter sample and one PreTrainNet update |
| DRCC | `x2g1s10` fixture, group 0; one parameter sample and one FullNet update |

These checks exercise model construction, original-region optimization,
polytope optimization, loss construction, backpropagation, and an optimizer
step. They do not run the complete paper schedules and do not write artifacts.

### 3. Run one update with the public three-phase inputs

```bash
python -m Simulator.runners.main_ds_3phase --smoke --no-save --device cpu --seed 0
```

This command loads both de-identified workbooks, constructs
`case36real_3phase_ds`, builds the original three-phase network model, and runs
one PreTrainNet update. `--no-save` makes the no-write intent explicit.

### 4. Optional geometry smoke checks

```bash
python -m Simulator.runners.main_polygon --smoke --no-save --device cpu --seed 0
python -m Simulator.runners.main_ellipse --smoke --no-save --device cpu --seed 0
python -m Simulator.runners.main_nonconvex --smoke --no-save --device cpu --seed 0
python -m Simulator.runners.main_cube --smoke --no-save --device cpu --seed 0
python -m Simulator.runners.main_ball --smoke --no-save --device cpu --seed 0
```

## Output behavior

The nine documented training runners default to the existing project
`results` directory:

- `main_polygon`, `main_ellipse`, `main_nonconvex`, `main_cube`, and
  `main_ball`;
- `main_agg`;
- `main_ds` and `main_ds_3phase`; and
- `main_drcc`.

This preserves the repository's established directory layout. A normal saved run
can replace an existing artifact with the same path and filename. Use
`--smoke --no-save` when checking executability without changing reference
results. `--output-root PATH` remains available for an explicitly chosen
alternative root; omit it to use the project structure documented here.

## Run the paper workflows

Full paper schedules can be computationally expensive. Inspect `--help` and use
`--case`, `--group`, `--dimensions`, or shortened epoch options before launching
a large sweep.

### Supplementary geometry cases

The 2-D runners support `pretrainnet`, `biasnet`, and `fullnet`:

```bash
python -m Simulator.runners.main_polygon --model-type pretrainnet --seed 0
python -m Simulator.runners.main_ellipse --model-type pretrainnet --seed 0
python -m Simulator.runners.main_nonconvex --model-type pretrainnet --seed 0
```

The hypercube and unit-ball studies use PreTrainNet. The dimensions reported in
Supplementary Note 2 range from 2 to 200:

```bash
python -m Simulator.runners.main_cube --dimensions 2 4 6 8 10 --seed 0
python -m Simulator.runners.main_ball --dimensions 2 4 6 8 10 --seed 0
```

Omitting `--dimensions` runs the configured paper sweep.

### Resource aggregation

Supplementary Note 3 defines two 24-period scenarios:

| CLI scenario | Resource population | Learned polytope | Paper schedule |
|---|---|---:|---|
| `continuous` | 600 continuously controlled EVs and 400 continuously controlled HPs | 96 inequalities in 24 dimensions | 500 + 200 + 100 updates; learning rate `0.02` |
| `mixed` | 54 continuous EVs, 36 continuous HPs, 6 on/off EVs, 4 on/off HPs, and 5 BSSs | 96 inequalities in 24 dimensions | 500 + 200 + 100 updates; learning rate `0.01` |

```bash
python -m Simulator.runners.main_agg --scenario continuous --seed 0
python -m Simulator.runners.main_agg --scenario mixed --seed 0
```

The code-level optimality-to-feasibility loss ratios are `1`, `0.1`, and
`1e-4` across the three phases, progressively prioritizing feasibility. The
runner performs exactly 800 updates.

The established result layout is:

- continuous: `results/aggregation`;
- mixed: `results/aggregation/discrete`; and
- Box comparison: `results/aggregation/data_cube`.

### Balanced distribution networks

Supplementary Note 4 uses eight balanced single-phase distribution cases:

```text
case10ba_ds       case17me_ds       case33bw_ds       case51ga_ds
case74_ds         case118zh_ds      case136ma_ds      case533mt_hi_ds
```

Pretrain a fixed polytope:

```bash
python -m Simulator.runners.main_ds --case case10ba_ds --model-type pretrainnet --seed 0
```

Train the two FullNet variants:

```bash
python -m Simulator.runners.main_ds --case case10ba_ds --model-type fullnet --variant feasible --seed 0
python -m Simulator.runners.main_ds --case case10ba_ds --model-type fullnet --variant moderate --seed 0
```

Omit `--case` to process all eight cases, or repeat it to select several.
The varying root-node voltage dataset contains 100 samples over
`[0.95, 1.05]` p.u.

The paper schedules are:

- PreTrainNet: 500 updates with learning rate `0.1 / p_base`;
- Moderate FullNet: 60 epochs with relative optimality weight `0.6`; and
- Feasible FullNet: 40 balancing epochs followed by 10
  feasibility-focused epochs, with case-adaptive learning rates specified in
  Supplementary Note 4.

FullNet loads `pretrainnet_weights.pth` from the selected output root and falls
back to the corresponding weight in the project's `results/ds_proj` directory.

### Real three-phase distribution network

The ninth distribution case is the 36-node, unbalanced three-phase network listed
as case `i` in Supplementary Table “Distribution network configurations.” Its
code identifier remains:

```text
case36real_3phase_ds
```

Keeping this identifier is intentional: it preserves the correspondence among
the manuscript, supplementary table, code, and the 27 T–D result combinations.
It is not a retained organization, customer, feeder, or location identifier.

Train PreTrainNet or the feasible FullNet variant:

```bash
python -m Simulator.runners.main_ds_3phase --model-type pretrainnet --seed 0
python -m Simulator.runners.main_ds_3phase --model-type fullnet --variant feasible --seed 0
```

The moderate variant is also available:

```bash
python -m Simulator.runners.main_ds_3phase --model-type fullnet --variant moderate --seed 0
```

The same 8-facet, two-dimensional interface-power polytope and
`[0.95, 1.05]` p.u. voltage sampling range are used as for the balanced cases.

### DRCC portfolio optimization

The four main cases in Fig. 4 and Supplementary Note 5 are:

| CLI case | Assets `N` | Groups `G` | Historical samples per asset `K` | Polytope inequalities per group |
|---|---:|---:|---:|---:|
| `50x2x150` | 50 | 2 | 150 | `4N_g + 2` |
| `150x3x300` | 150 | 3 | 300 | `4N_g + 2` |
| `300x5x900` | 300 | 5 | 900 | `4N_g + 2` |
| `400x8x1280` | 400 | 8 | 1,280 | `4N_g + 2` |

Each asset group varies four parameters:
`(epsilon, rho, R_min, X_max)`. Supplementary Note 5 specifies direct FullNet
training for 30 epochs with 100 parameter samples, batch size 5, two directional
calculations per sample, and Adam at learning rate `2e-4`.

Train one zero-based group:

```bash
python -m Simulator.runners.main_drcc --case 50x2x150 --group 0 --seed 0
```

Omit `--group` to train every group, or repeat it:

```bash
python -m Simulator.runners.main_drcc --case 400x8x1280 --group 0 --group 1 --seed 0
```

## Data inventory and provenance

### Aggregation

- `data/aggregator_data/ZH_buildings.csv` contains 340,586 Zurich building
  records. The model uses `HBLD`, `CBLD`, and `PRT` to parameterize HPs.
- `data/profiles_data/profiles_data.npz` contains aligned `load_data`,
  `pv_data`, and `temp_data` arrays, each of length 2,881.
- `data/profiles_data/CH_2021_real.mat` contains hourly temperature data for
  440 stations.
- `BL_samples.mat` and `PV_samples.mat` each contain a `96 x 10` sample
  matrix.

One building record contains infinite values in the unused coordinate columns
`GKODN` and `GKODE`; the aggregation model does not read those columns. The
complete deterministic preprocessing program that created
`profiles_data.npz` from the raw MAT files is not present, so the supplied NPZ is
the canonical processed input for exact reruns.

### Balanced T–D data

`data/TD_OPF/ds_data` contains the eight balanced distribution MAT files.
`case118_ts.mat` and `case300_ts.mat` provide two transmission cases; the
paper-specific `case4gs_ts` values are defined in
`Simulator/cases/TD_case.py`.

`data/TD_OPF/case4gsts.mat` is retained as a legacy file but is not loaded by the
documented workflow. Several of its generator values differ from the
paper-specific hard-coded case. Duplicate MAT/MATPOWER files outside `ds_data`
are retained for provenance and are not the canonical inputs.

### Public three-phase data and de-identification

The public three-phase inputs are:

```text
data/real_dis_data/load_file.xls
data/real_dis_data/volt_file.xls
```

`load_file.xls` contains 28 worksheets in the fixed order
`Load_Point_01` through `Load_Point_28`. The node-to-load mapping in
`DS_case_3phase.py` depends on this order; do not reorder the sheets. Across the
28 worksheets, the workbook contains 83,511 data records after excluding header
rows and trailing format-only rows.

Its 24 English columns are:

```text
management_unit
service_center
customer_id
customer_name
timestamp
asset_id
instantaneous_active_power
phase_a_current
phase_b_current
phase_c_current
neutral_current
phase_a_voltage
phase_b_voltage
phase_c_voltage
total_power_factor
cumulative_forward_active_energy
cumulative_reverse_active_energy
quadrant_i_reactive_energy
quadrant_iv_reactive_energy
ct_ratio
pt_ratio
logical_address
is_supplemental_reading
ingestion_timestamp
```

`volt_file.xls` contains 26,532 data records in worksheet
`Feeder_Measurements`, with:

```text
timestamp
feeder_active_power
feeder_reactive_power
feeder_current
bus_line_voltage_ab
```

The public workbooks were transformed as follows:

- worksheet names, column headings, and textual values were converted to English;
- organization, service-center, customer, asset, and logical-address identifiers
  were replaced by stable neutral identifiers;
- all timestamps retain their original 2024 calendar dates and times, row order,
  and temporal spacing;
- the selected operating instant is `2024-09-30 14:00`;
- electrical measurements, network topology, worksheet order, and the
  node-to-load correspondence were not changed; and
- `Load_Point_10` is the one worksheet whose phase currents are used without
  the `120` multiplier; the other 27 worksheets retain the original multiplier
  rule.

These transformations remove direct identifiers without changing the numerical
inputs used to construct the paper's 36-node case. No source-to-neutral identifier
mapping is included in this repository.

### DRCC

Each main DRCC case has a training CSV and a matching out-of-sample test CSV:

```text
r_samples_x{N}g{G}s{K}.csv
r_samples_x{N}g{G}s{K}_test.csv
```

The main training files have `N` rows, `K` sample columns, balanced group labels,
and a `mean_r` column consistent with the sample mean. Each test file contains
100 return samples per asset. The checked values are finite and lie in
`[-0.1, 0.1]`.

The generator refuses to replace an existing CSV unless replacement is explicitly
enabled. Record the random seed when generating a different dataset.

## Reference results

### Geometry and DRCC summary workbooks

The English summary filenames are:

```text
results/cube/summary_table.xlsx
results/ball/summary_table.xlsx
results/DRCC/parameter_settings.xlsx
```

### Aggregation

The established result layout is:

- continuous scenario: `results/aggregation/figures`,
  `results/aggregation/error_history.pkl`, and
  `results/aggregation/pretrainnet_weights.pth`;
- mixed scenario: `results/aggregation/discrete/figures`,
  `results/aggregation/discrete/error_history.pkl`, and the historical weight
  filename `results/aggregation/pretrainnet_weights.pthdisc`; and
- Box comparison: `results/aggregation/data_cube` and
  `results/aggregation/error_history_cube.pkl`.

Both scenarios contain 17 matching `A_i.csv`/`b_i.csv` snapshots for steps
0–800. Their final mean feasibility errors are `7.1238e-15` for the continuous
scenario and `1.0783e-5` for the mixed scenario, matching the rounded values in
Fig. 2. The `iterations` field in the two historical `error_history.pkl` files
is all zeros; the snapshot filenames are the authoritative step indices.

### T–D network optimization

- `results/ds_proj` contains the **Feasible PolyFormer** results.
- `results/ds_proj_original` contains the **Moderate PolyFormer** results.

Each `td_results/td_experiment_results.xlsx` workbook contains:

- `ds_case_parameters`: the nine distribution-case parameters;
- `pretrainnet`: fixed-polytope results;
- `fullnet`: the complete T–D evaluation used for the paper; and
- `Sheet1`: an additional retained summary sheet.

The `fullnet` sheet contains exactly 27 unique combinations: three transmission
systems times nine distribution systems.

| Result set | Mean maximum feasibility error | Maximum feasibility error | Mean signed objective error |
|---|---:|---:|---:|
| Feasible (`ds_proj`) | `6.3587e-10` | `3.7244e-9` | `7.2325e-4` |
| Moderate (`ds_proj_original`) | `2.6037e-6` | `3.0765e-5` | `3.7100e-5` |

The related English vertex workbooks in both `td_results` directories are:

```text
vertex_results_root_voltage_unfixed.xlsx
vertex_results_root_voltage_varying.xlsx
vertex_results_root_voltage_fixed.xlsx
```

`fullnet.xlsx`, `fullnet533.xlsx`, and `fullnet_3phase.xlsx` are retained
intermediate batch outputs; `td_experiment_results.xlsx` is the complete
paper-level workbook.

### DRCC

The four main result directories are:

```text
results/DRCC/x50g2s150
results/DRCC/x150g3s300
results/DRCC/x300g5s900
results/DRCC/x400g8s1280
```

Every expected group directory contains `fullnet_weights.pth`, and each main case
contains a 300-strategy `test_result.pkl`. PreTrainNet weights are intentionally
absent because Supplementary Note 5 initializes an outer polytope and trains
FullNet directly.

### Non-main historical artifacts

`results/MPC`, `results/safe_region`, `results/epigraph`, and
`results/ev_agg` are exploratory or historical workflows rather than evidence for
the paper's principal claims. The incomplete `results/DRCC/x210g3s420`,
`results/DRCC/x400g5s1280`, and `results/epigraph_specified` directories are also
outside the four main application cases.

## Implementation notes

- `ErrorCalculator` solves the original and approximating regions along sampled
  directions, identifies boundary points and active facets, and returns the
  geometry used by the differentiable loss.
- `Trainer.train(n_train=N)` performs exactly `N` epochs in serial and parallel
  modes.
- Final batches use their actual size. Gradient logging tolerates parameters
  without gradients and zero-length parameter blocks.
- The documented runners seed NumPy and PyTorch. Solver parallelism and GPU
  kernels can still cause small platform-dependent numerical differences.
- Canonical paths are rooted through `Simulator.PROJECT_ROOT` and do not depend
  on the caller's working directory after the module is launched from the project
  root.
- Original-region evaluation requires the corresponding optimization solver and
  input data even when trained neural weights are already available.

## Citation

Please cite the accompanying manuscript:

> Yilin Wen, Yi Guo, Bo Zhao, Wei Qi, Zechun Hu, Colin Jones, and Jian Sun.
> “Learning efficient representations of complex constraints for scalable
> optimization.” Manuscript.
