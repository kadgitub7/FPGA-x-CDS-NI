# FPGA-Accelerated Cardiac Arrhythmia Diagnosis System

An FPGA implementation of the CDS-NI (Cognitive Dynamic System with Natural Intelligence) algorithm for real-time cardiac arrhythmia detection from 12-lead ECG data. The system trains on the UCI Arrhythmia Dataset (452 patients, 279 features), converts the model to fixed-point arithmetic, and runs inference entirely on a Basys 3 FPGA (Xilinx Artix-7) communicating over UART.

The FPGA processes **4 patients in parallel** using shared model BRAMs and replicated per-patient datapaths.

---

## Here is a quick demo video to understand how the project works
[![Watch the video](https://img.youtube.com/vi/AOe1tF_W6hg/maxresdefault.jpg)](https://www.youtube.com/watch?v=AOe1tF_W6hg)
## How It Works

```
 12-lead ECG Data (279 features per patient)
              |
   ┌──────────┴──────────┐
   │  PYTHON (training)  │
   │                     │
   │  Algorithm 1: Build decision tree (ID3, multi-level, sex-split root)
   │  Algorithm 2: Compute healthy ranges + perceptor/executive weights
   │  Algorithm 3: Prune redundant actions, retain best per (node, disease)
   │  Algorithm 4: Fixed-point inference (golden model for verification)
   │                     │
   │  Export: 7 x .mem files (BRAM initialization for FPGA)
   └──────────┬──────────┘
              |
   ┌──────────┴──────────┐
   │   FPGA (inference)  │
   │                     │
   │  UART RX → 4 sensor_interfaces (load 4 patients)
   │         → tree_traversal (scan 215 nodes, 4 parallel comparisons)
   │         → af_engine (12 diseases × N actions, shared BRAM reads,
   │                      4 parallel accumulators + range comparators)
   │         → result_sender → UART TX (4 × 5-byte responses)
   │                     │
   │  Decision: HEALTHY / UNHEALTHY / SCREENING  (per patient)
   └─────────────────────┘
```

---

## Project Structure

```
FPGA-x-CDS-NI/
│
├── CDS_NI_Algorithms/              ← Python: ML algorithms (training)
│   ├── data/
│   │   ├── arrhythmia.data         ← UCI dataset (452 patients × 280 cols)
│   │   └── arrhythmia.names        ← Dataset documentation
│   ├── build_decision_tree.py      ← Algorithm 1: decision tree construction
│   ├── action_normalRange.py       ← Algorithm 2: healthy ranges + weights
│   ├── action_pruning.py           ← Algorithm 3: action refinement
│   └── decision_pipeline.py        ← Algorithm 4: floating-point inference
│
├── FixedPoint_pipeline/             ← Python: fixed-point conversion layer
│   ├── decision_pipeline_fixedPoint.py  ← Algorithm 4 in Q-format arithmetic
│   └── parameter_export.py         ← Exports trained model → .mem files
│
├── FPGA_Files/                      ← Verilog: FPGA implementation
│   └── CDS-NI_files/
│       ├── Design_Sources/          ← 14 synthesizable Verilog modules
│       │   ├── cds_top.v            ← Top-level: 4-lane master FSM + dispatcher
│       │   ├── sensor_interface.v   ← BRAM: stores 279 features per patient
│       │   ├── tree_traversal.v     ← FSM: scans 215 nodes, 4 parallel matches
│       │   ├── af_engine.v          ← FSM: computes Assurance Factor (4-lane)
│       │   ├── model_rom.v          ← 7 BRAMs: tree, actions, probabilities, ranges
│       │   ├── fixedMultiply.v      ← Q s1.15 × Q s1.15 → Q s2.30 (DSP slice)
│       │   ├── fixedDivide.v        ← Multiply-by-reciprocal, 2-stage pipeline
│       │   ├── af_accumulator.v     ← AF += delta with saturation clamping
│       │   ├── rangeComparator.v    ← Combinational: value outside [bmin, bmax]?
│       │   ├── decision_logic.v     ← Latches final decision on trigger
│       │   ├── result_sender.v      ← Serializes 5-byte result to UART TX
│       │   ├── rl_action_selector.v ← RL-based action ranking
│       │   ├── uart_rx.v            ← UART receiver (115200 baud)
│       │   └── uart_tx.v            ← UART transmitter (115200 baud)
│       ├── Simulation_Sources/      ← 5 Verilog testbenches
│       ├── Constraint_Sources/
│       │   └── cds_top.xdc          ← Basys 3 pin mapping
│       └── Diagrams/                ← FSM state diagrams (af_engine, tree_traversal)
│
├── fpga_uart_validator.py           ← Python: 10-fold CV + FPGA validation (5 metrics)
│
├── fpga_cv_output/                  ← Generated: 10-fold cross-validation output
│   ├── fold_0/ ... fold_9/          ← Each fold contains:
│   │   ├── *.mem                    ←   7 model .mem files for that fold
│   │   ├── test_vectors.mem         ←   test user stimulus
│   │   ├── expected_output.mem      ←   golden predictions (hex, for Verilog TB)
│   │   ├── golden_results.csv       ←   golden predictions (CSV, for FPGA mode)
│   │   └── af_trace.mem             ←   step-by-step AF trace (RTL debugging)
│   ├── cv_summary.json              ← Aggregate accuracy across all folds
│   └── cv_manifest.txt              ← Summary of fold contents
│
├── FixedPointAnalysis/              ← Generated: single-model export + analysis
│   ├── fpga_mem/                    ← .mem files trained on full dataset
│   ├── QFormatBit.txt               ← Q-format bit budget definitions
│   └── VariableRangeData.txt        ← Min/max/mean analysis of all variables
│
├── FPGA-Real-Time-Monitor/          ← Standalone: live interactive diagnosis
│   ├── fpga_realtime_monitor.py     ← Interactive CLI (setup + diagnose)
│   ├── fpga_mem/                    ← .mem files trained on all 452 users
│   ├── patient_template.csv         ← CSV template with 279 feature headers
│   ├── FPGA_Modules/                ← Complete copy of Verilog design
│   │   ├── Design_Sources/          ←   14 modules (4-lane parallel versions)
│   │   └── Constraint_Sources/      ←   Basys 3 pin mapping
│   └── README.md                    ← Setup + usage guide for real-time mode
│
├── FPGA_Technical_Deep_Dive.txt     ← Detailed technical documentation
├── PROJECT_TIMELINE.md              ← Development milestones
└── README.md                        ← This file
```

### Which .mem files are which?

| Location | Trained on | Purpose |
|----------|-----------|---------|
| `fpga_cv_output/fold_N/` | 90% of data (fold N's training set) | Validation: verify FPGA matches Python |
| `FixedPointAnalysis/fpga_mem/` | Full dataset (all 452 users) | Reference export for analysis |
| `FPGA-Real-Time-Monitor/fpga_mem/` | Full dataset (all 452 users) | Live diagnosis deployment |

---

## Getting Started

### Prerequisites

- **Python 3.8+** with `numpy`, `pandas`, `pyserial`
- **Xilinx Vivado** (tested with 2023.x)
- **Digilent Basys 3** FPGA board (Artix-7 XC7A35T)

### Option A: Validate FPGA Correctness (10-Fold CV)

This trains 10 models, runs Python golden inference, exports `.mem` files per fold,
then sends test users to the FPGA and compares results with 5 metrics.

**Step 1 — Generate golden data + .mem files (run once)**

```bash
python fpga_uart_validator.py --mode software
```

**Step 2 — Load fold 0 into Vivado**

1. Create a Vivado project with all 14 `.v` files from `FPGA_Files/CDS-NI_files/Design_Sources/`
2. Add `FPGA_Files/CDS-NI_files/Constraint_Sources/cds_top.xdc`
3. Copy the 7 `.mem` files from `fpga_cv_output/fold_0/` into the Vivado project directory

**Step 3 — Synthesize + program the FPGA**

Run Synthesis → Implementation → Generate Bitstream → Program Device.

**Step 4 — Run FPGA validation**

```bash
python fpga_uart_validator.py --mode fpga --fold 0 --port COM3
```

Sends ~46 test users to the FPGA in batches of 4 and prints a 5-metric comparison report:

1. **Bit-Exact Match Rate** — FPGA decision vs Python decision (binary: healthy/unhealthy)
2. **AF Value Deviation** — mean/max difference in Assurance Factor
3. **Per-User Latency** — UART round-trip timing with estimated FPGA compute time
4. **Throughput** — users/sec comparison (FPGA vs Python)
5. **Binary Confusion Matrix** — accuracy, sensitivity, specificity for both

### Option B: Real-Time Interactive Diagnosis

This trains on all 452 users and provides a live CLI for diagnosing patients.

**Step 1 — Train + export**

```bash
cd FPGA-Real-Time-Monitor
python fpga_realtime_monitor.py --setup
```

**Step 2 — Load into Vivado**

Use the Verilog files from `FPGA-Real-Time-Monitor/FPGA_Modules/` and the `.mem` files from `FPGA-Real-Time-Monitor/fpga_mem/`.

**Step 3 — Synthesize + program the FPGA**

**Step 4 — Run**

```bash
python fpga_realtime_monitor.py --run --port COM3
```

```
CDS> diagnose 42                  # diagnose 1 patient from dataset
CDS> diagnose 10 20 30 40         # diagnose 4 patients in parallel
CDS> load patient_data.csv
CDS> diagnose me                  # diagnose all patients from CSV
CDS> diagnose all                 # run all 452 users through FPGA
```

See `FPGA-Real-Time-Monitor/README.md` for full details.

---

## 4-Lane Parallel Architecture

The FPGA processes 4 patients simultaneously. Model BRAMs are read once per action
and the results are broadcast to all 4 lanes. Only per-patient state is replicated:

| Component | Instances | Why |
|-----------|-----------|-----|
| model_rom (7 BRAMs) | 1 shared | Model parameters are the same for all patients |
| fixedMultiply | 1 shared | delta_AF depends only on model params, not sensor data |
| fixedDivide | 1 shared | Same reason |
| sensor_interface | 4 copies | Each patient has different feature values |
| af_accumulator | 4 copies | Each patient accumulates its own AF |
| rangeComparator | 4 copies | Each patient's values compared against same range |

**Basys 3 resource budget:**

| Resource | Available | Used | Headroom |
|----------|-----------|------|----------|
| BRAM18 | 100 | ~49 | 51 |
| DSP48 | 90 | 1 | 89 |
| LUTs | 20,800 | ~1,200 | ~19,600 |

---

## Dataset

UCI Cardiac Arrhythmia Database: 452 patients, 279 features from 12-lead ECG recordings.

- **Features 0-14**: Age, Sex, Height, Weight, QRS duration, PR/QT/T/P intervals, heart rate
- **Features 15-278**: 12 ECG channels (DI, DII, DIII, AVR, AVL, AVF, V1-V6), each with wave widths, deflections, and amplitudes
- **Label**: 1 = healthy, 2-16 = 15 arrhythmia classes

The original algorithms credit goes to the following. If you want to learn more about the CDS, NI algorithms see: 

Naghshvarianjahromi, Mahdi, Shiva Kumar, and M. Jamal Deen. "Brain-inspired intelligence for real-time health situation understanding in smart e-health home applications." IEEE Access 7 (2019): 180106-180126.
---

## Key Files Reference

| File | What it does |
|------|-------------|
| `fpga_uart_validator.py` | Main validation script: 10-fold CV + FPGA comparison |
| `CDS_NI_Algorithms/build_decision_tree.py` | Algorithm 1: builds multi-level decision tree |
| `CDS_NI_Algorithms/action_normalRange.py` | Algorithm 2: computes healthy ranges + weights |
| `CDS_NI_Algorithms/action_pruning.py` | Algorithm 3: prunes and refines action library |
| `FixedPoint_pipeline/decision_pipeline_fixedPoint.py` | Algorithm 4: fixed-point inference (golden model) |
| `FixedPoint_pipeline/parameter_export.py` | Converts trained model → 7 `.mem` files for FPGA |
| `FPGA_Files/.../cds_top.v` | FPGA top-level: 4-lane master FSM |
| `FPGA_Files/.../af_engine.v` | FPGA core: Assurance Factor computation (4-lane) |
| `FPGA_Files/.../tree_traversal.v` | FPGA: decision tree node scanning (4-lane) |
| `FPGA_Files/.../model_rom.v` | FPGA: all 7 model BRAMs |
| `FPGA-Real-Time-Monitor/fpga_realtime_monitor.py` | Interactive real-time diagnosis CLI |
