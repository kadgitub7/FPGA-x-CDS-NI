# FPGA Real-Time Cardiac Diagnosis Monitor

Real-time cardiac arrhythmia diagnosis system using FPGA hardware acceleration.
Unlike the validation pipeline in `FPGA-x-CDS-NI` (which uses 10-fold cross-validation
on a fixed dataset), this system trains on **all 452 patients** and provides
**live interactive diagnosis** via UART.

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
│   ├── prob_pgt1.mem
│   └── cds_params.vh
├── FPGA_Modules/
│   ├── Design_Sources/         # All 14 Verilog modules (copied from FPGA-x-CDS-NI)
│   └── Constraint_Sources/     # Pin mapping for Basys 3
└── README.md
```

## Quick Start

### 1. Setup (train model, export .mem files)

```bash
python fpga_realtime_monitor.py --setup
```

This trains Algorithms 1-3 on all 452 users and exports `.mem` files to `fpga_mem/`.

### 2. Synthesize FPGA

1. Create a Vivado project with the Verilog files from `FPGA_Modules/Design_Sources/`
2. Add the constraint file from `FPGA_Modules/Constraint_Sources/`
3. Copy all `.mem` files from `fpga_mem/` into your Vivado project
4. Synthesize, implement, and program the Basys 3 FPGA

### 3. Run Real-Time Diagnosis

```bash
python fpga_realtime_monitor.py --run --port COM3
```

### 4. Interactive Commands

At the `CDS>` prompt:

| Command | Description |
|---------|-------------|
| `diagnose me` | Diagnose patient from a loaded CSV file |
| `diagnose <id>` | Diagnose user #id from the arrhythmia dataset |
| `diagnose all` | Run all 452 users through the FPGA |
| `load <path.csv>` | Load a patient CSV file for `diagnose me` |
| `list` | Show all users in the dataset |
| `features` | Show the 279 ECG feature names |
| `quit` | Exit |

## Patient CSV Format

The `patient_template.csv` file (generated during setup) contains the 279 ECG feature
column names. Fill in the patient's measurements:

- **Columns 0-14**: Age, Sex (0=M/1=F), Height, Weight, QRS duration, PR interval, etc.
- **Columns 15-278**: 12-lead ECG measurements (DI through V6, wave/amplitude features)
- Missing values: use `?` or leave blank

## Diagnosis Output

The FPGA returns one of three decisions:

- **HEALTHY**: No arrhythmia detected
- **UNHEALTHY**: Arrhythmia detected, with specific disease class (2-16)
- **SCREENING**: Inconclusive, further tests recommended

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

## Difference from Validation Pipeline

| Aspect | FPGA-x-CDS-NI (Validation) | FPGA-Real-Time-Monitor |
|--------|----------------------------|------------------------|
| Training | 10-fold CV (90% train, 10% test) | All 452 users |
| Purpose | Verify FPGA matches Python model | Live patient diagnosis |
| Output | Match rate, accuracy metrics | Real-time diagnosis per patient |
| .mem files | 10 sets (one per fold) | 1 set (trained on all data) |
| Interaction | Batch processing | Interactive CLI |
