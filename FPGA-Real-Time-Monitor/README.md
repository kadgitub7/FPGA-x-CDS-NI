# FPGA Real-Time Cardiac Diagnosis Monitor

Real-time cardiac arrhythmia diagnosis system using FPGA hardware acceleration
with **4-lane parallel processing** — diagnoses up to 4 patients simultaneously.

Unlike the validation pipeline in `FPGA-x-CDS-NI` (which uses 10-fold cross-validation
on a fixed dataset), this system trains on **all 452 patients** and provides
**live interactive diagnosis** via UART.

## 4-Lane Parallel Architecture

The FPGA processes 4 patients in parallel using shared model BRAMs and replicated
per-patient datapaths:

```
         UART RX
            |
      [Dispatcher FSM]
       /   |   |   \
   sensor  sensor  sensor  sensor    (4 copies — each stores 279 features)
      0      1      2      3
       \   |   |   /
     [Tree Traversal]                (1 shared — broadcasts address to all 4 sensors,
      4 comparisons per node)         produces 4 match signals per node)
       \   |   |   /
       [AF Engine]                   (1 shared FSM — reads model BRAMs once per action,
      4 accumulators                  4 range comparators run in parallel,
      4 range comparators             1 shared multiplier + divider)
       /   |   |   \
   result  result  result  result    (4 decisions sent sequentially over UART)
      0      1      2      3
            |
         UART TX
```

**What is shared (1 copy):** model_rom (all 7 BRAMs), fixedMultiply, fixedDivide, tree traversal FSM, af_engine FSM.

**What is replicated (4 copies):** sensor_interface, af_accumulator, rangeComparator, node-match bitmaps, decision/AF/alarm registers.

When fewer than 4 patients are sent, the remaining lanes are padded with duplicate
data and the extra results are discarded.

## Directory Structure

```
FPGA-Real-Time-Monitor/
├── fpga_realtime_monitor.py    # Main Python script (setup + real-time diagnosis)
├── patient_template.csv        # Generated during setup — fill in patient features
├── fpga_mem/                   # Generated .mem files for FPGA BRAMs
│   ├── tree_topology.mem
│   ├── hr_index.mem
│   ├── hr_pairs.mem
│   ├── action_hdr.mem
│   ├── action_data.mem
│   ├── prob_phf.mem
│   └── prob_pgt1.mem
├── FPGA_Modules/
│   ├── Design_Sources/         # 14 Verilog modules (4-lane parallel versions)
│   │   ├── cds_top.v           # 4-lane master FSM + dispatcher
│   │   ├── tree_traversal.v    # 4 sensor inputs, 4 match outputs
│   │   ├── af_engine.v         # 4 accumulators, 4 range comparators, shared math
│   │   ├── sensor_interface.v  # Unchanged (instantiated 4 times)
│   │   ├── model_rom.v         # Unchanged (1 shared instance)
│   │   ├── af_accumulator.v    # Unchanged (instantiated 4 times in af_engine)
│   │   ├── rangeComparator.v   # Unchanged (instantiated 4 times in af_engine)
│   │   ├── fixedMultiply.v     # Unchanged (1 shared instance in af_engine)
│   │   ├── fixedDivide.v       # Unchanged (1 shared instance in af_engine)
│   │   ├── decision_logic.v    # Unchanged (called 4 times sequentially)
│   │   ├── result_sender.v     # Unchanged (called 4 times sequentially)
│   │   ├── rl_action_selector.v
│   │   ├── uart_rx.v           # Unchanged
│   │   └── uart_tx.v           # Unchanged
│   └── Constraint_Sources/
│       └── cds_top.xdc         # Pin mapping for Basys 3
└── README.md
```

## Quick Start

### 1. Setup (train model, export .mem files)

```bash
python fpga_realtime_monitor.py --setup
```

Trains Algorithms 1-3 on all 452 users and exports `.mem` files to `fpga_mem/`.

### 2. Synthesize FPGA

1. Create a Vivado project with the Verilog files from `FPGA_Modules/Design_Sources/`
2. Add the constraint file from `FPGA_Modules/Constraint_Sources/`
3. Copy the 7 `.mem` files from `fpga_mem/` into your Vivado project
4. Synthesize, implement, and program the Basys 3 FPGA

### 3. Run Real-Time Diagnosis

```bash
python fpga_realtime_monitor.py --run --port COM3
```

### 4. Interactive Commands

At the `CDS>` prompt:

| Command | Description |
|---------|-------------|
| `diagnose me` | Diagnose all patients from a loaded CSV (batched by 4) |
| `diagnose <id>` | Diagnose 1 user (padded to 4 lanes, extra results discarded) |
| `diagnose <id1> <id2> ...` | Diagnose 1-4 users in parallel |
| `diagnose all` | Run all 452 users through the FPGA (batches of 4) |
| `load <path.csv>` | Load a patient CSV file for `diagnose me` |
| `list` | Show all users in the dataset |
| `features` | Show the 279 ECG feature names |
| `quit` | Exit |

### Examples

```
CDS> diagnose 42                    # 1 patient (lanes 1-3 padded)
CDS> diagnose 10 20 30 40           # 4 patients in parallel
CDS> diagnose 5 100                 # 2 patients (lanes 2-3 padded)
CDS> load patient_data.csv
CDS> diagnose me                    # all rows from CSV, 4 at a time
CDS> diagnose all                   # all 452 dataset users, 4 at a time
```

## UART Protocol

The FPGA expects exactly 4 patients per batch:

**PC to FPGA:** 4 packets back-to-back, each `[0xAA header] [558 feature bytes]`

```
[0xAA][558 bytes patient 0][0xAA][558 bytes patient 1][0xAA][558 bytes patient 2][0xAA][558 bytes patient 3]
```

**FPGA to PC:** 4 result packets, each 5 bytes

```
[5 bytes result 0][5 bytes result 1][5 bytes result 2][5 bytes result 3]
```

Each 5-byte result: `{2'b00, alarm_class[3:0], decision[1:0]}`, `AF[31:24]`, `AF[23:16]`, `AF[15:8]`, `AF[7:0]`

When fewer than 4 patients are needed, the Python script pads the batch with
copies of the last real patient and discards the extra FPGA results.

## Patient CSV Format

The `patient_template.csv` file (generated during setup) contains the 279 ECG feature
column names. Fill in the patient's measurements:

- **Columns 0-14**: Age, Sex (0=M/1=F), Height, Weight, QRS duration, PR interval, etc.
- **Columns 15-278**: 12-lead ECG measurements (DI through V6, wave/amplitude features)
- Missing values: use `?` or leave blank
- Multiple patients: one row per patient (all diagnosed in batches of 4)

## Diagnosis Output

The FPGA returns one of two decisions per patient:

- **HEALTHY**: No arrhythmia detected (AF below threshold)
- **UNHEALTHY**: Arrhythmia detected (a feature fell outside the healthy range for a disease)

Disease classes:

| ID | Condition |
|----|-----------|
| 2 | Ischemic changes (Coronary Artery Disease) |
| 3 | Old Anterior Myocardial Infarction |
| 4 | Old Inferior Myocardial Infarction |
| 5 | Sinus Tachycardia |
| 6 | Sinus Bradycardia |
| 7 | Ventricular Premature Contraction |
| 8 | Supraventricular Premature Contraction |
| 9 | Left Bundle Branch Block |
| 10 | Right Bundle Branch Block |
| 14 | Left Ventricular Hypertrophy |
| 15 | Atrial Fibrillation or Flutter |
| 16 | Others |

## Requirements

- Python 3.8+
- numpy, pandas
- pyserial (for FPGA communication)
- Xilinx Vivado (for FPGA synthesis)
- Digilent Basys 3 FPGA board

## Estimated FPGA Resource Usage (Basys 3 XC7A35T)

| Resource | Available | Model BRAMs | Per lane | 4 lanes total | Headroom |
|----------|-----------|-------------|----------|---------------|----------|
| BRAM18 | 100 | ~45 | ~1 | ~49 | ~51 |
| DSP48 | 90 | 0 | 0 | 1 (shared) | 89 |
| LUTs | 20,800 | — | ~300 | ~1,200 | ~19,600 |
| FFs | 41,600 | — | ~300 | ~1,200 | ~40,400 |

## Difference from Validation Pipeline

| Aspect | FPGA-x-CDS-NI (Validation) | FPGA-Real-Time-Monitor |
|--------|----------------------------|------------------------|
| Training | 10-fold CV (90% train, 10% test) | All 452 users |
| Purpose | Verify FPGA matches Python model | Live patient diagnosis |
| Parallelism | 4 users per batch | 4 users per batch |
| Output | 5-metric comparison report | Real-time diagnosis per patient |
| .mem files | 10 sets (one per fold) | 1 set (trained on all data) |
| Interaction | Batch validation script | Interactive CLI |
