# PROJECT TIMELINE & OBJECTIVES

**Short Disclaimer**
This .md file is for people that want to understand what process as well as stuggles I went through to complete this project. Through this approach I will learn a lot and hit many roadblocks, which I hope to overcome. By following this file, you can see what my objectives, by path to implementation, and learning is.

## 0. Understanding the Algorithms
 a. Algorithm 1: build_decision_tree.py:
 - The first algorithm is used to build a decision tree. This is the basis on which the rest of the algorithm works. A simplified explanation is that it takes the data, sorts it into features and labels them. It builds a tree which at the root invludes all features and each child branch is formed based on the splitting of the feature(Binary or Continuous(through median)). Once this is done each branch is checked to see if it meets the minimum user requirement(around 200), if not it is pruned. for tiers in the branch that involve level 3 redundant branches are removed(branches where the same subset of population exists). Therefore the final output is a tree where different users are situated and organized based on their feature values. At the top there is the entire population and they are slowly filtered into smaller groups.

 b. Algorithm 2: action_normalRange.py
 - This algorithm goes through each feature in each node and computes the probabilties to be able to determine how likely people have a disease in the population, how likely a user that is unhealthy is ourside the range and how likely a person in this bin has a specific disease class and the such. One these are done we calculate the best actions by how likely unhealthy people are outside the normal range and rank them.

 c. Algorithm 3: action_pruning.py
 - In this algorithm we go through the action list. We prune any actions that have weight of 0. Then we go through and prune actions have do not provide any new information compared to previous actions. This makes it so that only the highest value actions are used.

 d. Algorithm 4: decision_pipeline.py
 - In this algorithm we build the decision tree, compute the probabilities, refine actions through the other algorithms. Once we have done sthat we go through a loop of deciding what is the best actions by running a simulation on the actions based on the probabilities in algorithm 2. It then chooses the action and computes the AF and rw based on that result. It also uses a hard boundary to seperate healthy vs unhealthy users. It repeates this step for each node, focus level and action until a user is classified as unhealthy for being outside a range, healthy for having enough AF, or screening for exhausting the entire tree.

## 1. Fixed-Point Analysis and Model Export
 a. Objective: Convert all floating-point computations in Algorithm 4 to fixed-point arithmetic and export trained model parameters to a format loadable by the FPGA.
 - Why: FPGAs don't have native floating-point units. Every multiplication, division, and comparison in Algorithm 4 must be expressed in fixed-point (e.g., Q16.16 or Q8.24 format). Choosing the wrong format causes either overflow (too few integer bits) or loss of precision (too few fractional bits) -- both silently corrupt results.

 First we must profile all the variables in Algorithm 4. This is the algorithm that will be implemented in FPGA architecture. Therefore we must know the range of the variables used here.

 Realized that the current LOOCV runs with too much time and computational cost as well as too many variables, changing to 10 - Fold Cross Validation: These are the results from switching:
 
 Users evaluated  : 452
  Overall accuracy : 71.5%
  Sensitivity      : 82.1%
  Specificity      : 62.4%
  False alarm rate : 37.6%
  Screening count  : 0
 

 b. Deciding the number of bits in Q format to assign to each intermediate value being processed in the FPGA. This is important because we want to minimize the numebr of bits used, but at the same time also make sure that there are enough bits to not affect the calcualtions and later the results. 

 I have attached the appended list of the bits needed for each variable and an explanation to the QFormatBit.txt.

 There are three main criterion which are used to determine the Q format:
 1. What is the WORST-CASE magnitude?  →  determines INTEGER bits
 2. Can it go NEGATIVE?                →  determines if you need a SIGN bit
 3. How SMALL can meaningful values get? →  determines FRACTIONAL bits

 In this project all signed values were used to make sure that there are no hidden bugs when trasitioning to verilog. This is because a mix of signed and unsigned values can be misrepresented in Verilog easily.

 c. Once I ran the fixed point algorithm side by side with the floating point algorithm we got the following results:
 
 Cross-Validation Fold Accuracy Comparison

| Fold | Fixed Point Accuracy | Floating Point Accuracy |
|------:|---------------------:|-------------------------:|
| 1 | 60.9% | 60.9% |
| 2 | 66.3% | 66.3% |
| 3 | 68.1% | 68.1% |
| 4 | 71.2% | 71.2% |
| 5 | 72.2% | 72.2% |
| 6 | 71.4% | 71.4% |
| 7 | 71.1% | 71.1% |
| 8 | 70.1% | 70.1% |
| 9 | 71.3% | 71.0% |
| 10 | 71.7% | 71.5% |

---

Overall Performance Metrics

| Metric | Fixed Point | Floating Point |
|:------------------|------------:|----------------:|
| Users Evaluated | 452 | 452 |
| Overall Accuracy | 71.7% | 71.5% |
| Sensitivity | 82.1% | 82.1% |
| Specificity | 62.9% | 62.4% |
| False Alarm Rate | 37.1% | 37.6% |
| Screening Count | 0 | 0 |

---

Per-Class Detection Performance

| Class | Total Samples | Fixed Point Detected | Fixed Point Rate | Floating Point Detected | Floating Point Rate |
|-------:|--------------:|---------------------:|-----------------:|------------------------:|--------------------:|
| 2 | 44 | 34 | 77.3% | 34 | 77.3% |
| 3 | 15 | 15 | 100.0% | 15 | 100.0% |
| 4 | 15 | 11 | 73.3% | 11 | 73.3% |
| 5 | 13 | 12 | 92.3% | 12 | 92.3% |
| 6 | 25 | 20 | 80.0% | 20 | 80.0% |
| 7 | 3 | 2 | 66.7% | 2 | 66.7% |
| 8 | 2 | 2 | 100.0% | 2 | 100.0% |
| 9 | 9 | 9 | 100.0% | 9 | 100.0% |
| 10 | 50 | 40 | 80.0% | 40 | 80.0% |
| 14 | 4 | 4 | 100.0% | 4 | 100.0% |
| 15 | 5 | 5 | 100.0% | 5 | 100.0% |
| 16 | 22 | 16 | 72.7% | 16 | 72.7% |

---

Summary

The fixed-point implementation produces nearly identical classification behavior to the floating-point implementation. Small deviations occur in later folds (Fold 9–10), producing:

- +0.2% overall accuracy for fixed point
- +0.5% specificity for fixed point
- −0.5% false alarm rate for fixed point

A single user was classified as healthy who was healthy instead of incorrectly unhealthy in the floating point interpretation. This could be due to a rounding error when converting to fixed point, but was favourable. This change is within the expected different and will not be changed.

## 2. Model Parameter Export
The FPGA has BRAM(Block RAM). This is where all the information is stored. We need to create a system where all the values processed in Algorithm 1-3 is stored in look up tables that can be easily accessed to get the relevant information.

We can store all of these values in .mem files which can store the values and be accessed on run time with the FPGA synthesis.


## 3. FPGA setup
I plan on running a small test file to test if the entire process of building the HDL, testing, and simulating the project will work properly.
If we assume that the clock frequency is 50 MHz and we want the LED to blink at 1Hz, then we need to find the number of bits in the counter to accomodate for that. The formula:

f(blinking) = f(clk)/(2^n+1)
1Hz = 50 MHz / 2^(n+1)
1Hz = 50,000,000 Hz  / 2^n+1
2^n+1 = 50,000,000
n+1 = log_2(50,000,000)
n+1 ~ 25.58
n ~ 24.58
n ~ 25

Here is the waveform extracted from the simulation of the LED 1 Hz blinking:

<img src=/FPGA_Files/waveforms/LED_Blinking_waveform.png width="40%">

Next stages(To verify everything in simulation properly):
Create Constraint file, Flow Navigator, Run Synthesis, Run Implementation, Generate Bitstream

If all work out well move onto the following:
Naviatage to the file, go to <file name>.runs and then go into impl_1 and search for <top file>.bit.

Then:
Plug in Your FPGA, Turn board power on, Open: Flow Navigator, Open Hardware Manager, Open Target, Auto Connect -> You should see the FPGA device. Then: Program Device, Select the generated .bit file.

Once this is done the FPGA will do what the programming says, here is a video of FPGA blinking LED: U16 at 1Hz, I also press button U18 to reset according to the constraints file:

[![Watch the video](https://img.youtube.com/vi/2KW8qRrUJhc/0.jpg)](https://www.youtube.com/shorts/2KW8qRrUJhc)

## 4. CDS-NI Module Coding

First developed the module for range comparison, this module was one of the more simpler ones as is was purely combinational.
The fixed mulitply and fixed divide were implemented next and were similar due to their combinational and simple sequential logic.
The af_accumulate module was a little more tricky and needed both combinational, sequential(clock based) methods.

The af_engine module is complex and represent a full flow of how a user may be checked with regards to assurance factor. It involves multiple of the previous modules and combines then in the form of a FSM(Finite State Machine). Below is a diagram drawn to represent the different states, and actions that transition the machine.

<img src=/FPGA_Files/CDS-NI_files/Diagrams/af_engine_FSM_Diagram.png width="60%">

The module is quite long with many components. It creates instances of all previous modules which it uses to multiply, divide, compare boundaries, and accumulate assurance. It runs the standards pipeline and accesses memory components. It holds many different states which each transisition based on stict logic.

The next module is for tree traversal. It identifies which node to compare the user on by taking their value and determining which branch they belong to, returning a stream of values. Here is the FSM diagram for this module:

<img src=/FPGA_Files/CDS-NI_files/Diagrams/tree_traversal_FSM_Diagram.png width="60%">

The tree traversal module is made to go through all the possible nodes mentioned in a memory file and iterate through them while determining if the user value falls within this range. It waits a few clock cycles to get the tree node address and the users value to compare. Once this is done it returns a stream of values which help us who which nodes to compare healthy ranges for in AF engine module.

The sensor_interface module was made to be a transition point where the value of the user can be accurately read by the tree traversal as well as the af engine module. It helps to take the users data from the uart stream and organize it to be used in later sections.

Created model_rom module which acts as a way to invoke parameterized modules which can create the BRAM memory elements from the .mem files. This allows us to take in the inputs that were generated from alg 1-3 without the use of libraries and inputs like in python.

decision logic was another module created which is an add on to the tree traversal module. It simple consolidates the final decision by receiving it from the tree traversal module and sends the final complete signal. Once this is done result sender module can go through the process of sending only the bits that are needed in the python validation step: the final decision, the af, and the class that the person belonged to.

The UART protocol modules where imported from NANDLAND source. This is because it is a standardized module which is not specific to the project. An understanding of what the module does it shown below:

- This is the input and output connecting the FPGA to the computer. This is how we send over the users sensor data from the computer to the FPGA as well as send the final decision results to the computer. We use multiple states as well as different port settings which are set beforehand so that the computer and FPGA know how fast the bits will come. The set time is then used to determine after the high signal is set low when to capture bits and store them on their respective devices.

When idle (no data), the wire sits at logic 1 (high). A byte is sent as a frame of 10 bits:

IDLE ─────┐   ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌───── IDLE
          │   │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
          └───┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘
          START  D0  D1  D2  D3  D4  D5  D6  D7  STOP
Start bit (always 0): Signals "a byte is coming." The falling edge from idle (1) to start (0) is how the receiver detects a new byte.
8 data bits (D0-D7): LSB first. To send 0x41 ('A' = 01000001), the wire carries: 1,0,0,0,0,0,1,0 (bit 0 first).
Stop bit (always 1): Signals "byte is done." Returns the line to idle state.
Each bit is held on the wire for exactly 1/baud_rate seconds. At 115200 baud, each bit lasts ~8.68 microseconds.

## 5. Top-Level Integration (cds_top.v)

This was the hardest part of the project by far. Taking all the individual modules and connecting them together under one master controller that actually works end-to-end. The cds_top module is a 14-state FSM that orchestrates the entire pipeline: receiving patient data over UART, running tree traversal, executing the AF engine on matched nodes, and sending results back.

The states are:
- S_IDLE to S_WAIT_LOAD: Waits for UART to finish loading a patient's 279 features into the sensor interface BRAM
- S_START_TREE to S_SCAN_TREE to S_CHECK_NODES to S_FIND_NODE: Runs tree traversal across all 215 nodes and collects which nodes the patient belongs to
- S_START_AF to S_WAIT_AF to S_CHECK_AF to S_NEXT_NODE: For each matched node, fires the AF engine which iterates through all disease classes and actions, computing the assurance factor
- S_OUTPUT →toS_WAIT_SEND to S_NEXT_SEND to S_DONE: Serializes the result (decision + AF + alarm class) back through UART TX

The biggest issue during integration was timing. Each sub-module (tree traversal, AF engine, model ROM) has its own internal latency BRAM reads take 1-2 clock cycles, the multiplier and divider have pipeline stages. Getting the handshake signals right between the master FSM and each module took a lot of debugging. I had to add wait states and careful signal sequencing to make sure data was actually valid before the next module consumed it.

Another challenge was the address muxing for the sensor interface. Both tree_traversal and af_engine need to read patient feature values from the same BRAM, but at different times. I used a mux controlled by the master FSM state (sensor_mux_sel) to switch the address bus between the two modules.

## 6. FPGA-Python Communication Pipeline

Once the Verilog was working in simulation, the next step was getting it to talk to the Python side over UART. This involved two things:

 a. The result_sender module on the FPGA side, it takes the final decision (2 bits), the accumulated AF value (32 bits), and the alarm class (4 bits) and serializes them into 5 bytes that get clocked out through the UART TX one byte at a time.

 b. The Python side (fpga_uart_validator.py), this script handles the full conversation: packing a patient's 279 features into bytes with a header byte (0xAA), sending them over the serial port, and then reading back the 5-byte response.

The protocol is straightforward: Python sends 0xAA followed by 558 bytes (279 features × 2 bytes each, big-endian), then waits for 5 bytes back. Getting this to work reliably was harder than expected: there were issues with timing between when the FPGA finished processing and when Python expected a response. I had to add proper synchronization so neither side got out of step.

## 7. Validation Framework (10-Fold Cross-Validation)

I built a full validation pipeline to prove the FPGA produces the same results as Python. The fpga_uart_validator.py script has two modes:

 a. Software mode (--mode software): Trains 10 separate models using 10-fold cross-validation. For each fold, it runs Algorithms 1-3 on the training set, then runs Algorithm 4 (fixed-point inference) on the test set as the golden model. It exports 7 .mem files per fold (for FPGA synthesis) plus golden_results.csv with the expected outputs.

 b. FPGA mode (--mode fpga --fold N): Takes the test users from fold N, sends them to the FPGA in batches of 4, reads back the results, and compares against the golden model with 5 metrics:
  1. Bit-Exact Match Rate does the FPGA decision match the Python decision for every user?
  2. AF Value Deviation how far off are the actual assurance factor numbers?
  3. Per-User Latency UART round-trip time with estimated FPGA compute portion
  4. Throughput users per second for FPGA vs Python
  5. Binary Confusion Matrix accuracy, sensitivity, specificity

The goal was 100% match rate the FPGA should produce identical decisions to the fixed-point Python model. Not "close enough", identical. If the fixed-point math is implemented correctly in Verilog, there should be zero deviation.

## 8. Scaling to 4-Lane Parallel Processing

Initially everything was single-lane one patient at a time. The move to 4 patients in parallel was a big architectural change. I had to think carefully about which hardware gets shared and which gets replicated.

The key insight: the model parameters (BRAMs for tree topology, actions, probabilities, healthy ranges) are identical for every patient. The multiply and divide operations depend only on model parameters, not on patient data. So those stay as single shared instances. What does change per patient is the sensor data, the AF accumulator, and the range comparator. Those get replicated 4 times.

In cds_top, I added a load_lane counter that cycles through lanes 0-3 during the UART loading phase. Each lane has its own sensor_interface BRAM. During tree traversal, all 4 sensor BRAMs are read with the same feature address simultaneously so 4 comparisons happen at no extra cost. Same thing in the AF engine: the multiplier and divider compute once, and the result gets broadcast to 4 accumulators and 4 range comparators.

The result: processing 4 patients takes essentially the same number of clock cycles as processing 1. The UART loading time is 4x longer (since we still have a single serial link), but the compute time stays nearly flat. If you replaced UART with something faster like SPI, the parallelism would actually matter a lot.

| Component | Instances | Why |
|-----------|-----------|-----|
| model_rom (7 BRAMs) | 1 shared | Model parameters are the same for all patients |
| fixedMultiply | 1 shared | delta_AF depends only on model params, not sensor data |
| fixedDivide | 1 shared | Same reason |
| sensor_interface | 4 copies | Each patient has different feature values |
| af_accumulator | 4 copies | Each patient accumulates its own AF |
| rangeComparator | 4 copies | Each patient's values compared against same range |

## 9. BRAM Optimization Two-Level Healthy Range Lookup

One problem I ran into during synthesis was the healthy range table. The direct mapping would be node_idx × 279 features × 2 values (bmin, bmax) = about 131,000 entries. On the Basys 3, each BRAM18 block holds around 1,024 entries at 16 bits. That would need over 100 BRAM blocks, but the Basys 3 only has 100 total and the rest of the model needs BRAMs too.

The solution was pair deduplication. Many nodes share the same healthy range boundaries for a given feature. So instead of storing the full table, I implemented a two-level lookup in model_rom:

 Level 1: An index table (hr_index.mem) that maps each (node, feature) pair to a pair_id. ~60,000 entries of 16-bit IDs.
 Level 2: A pair table (hr_pairs.mem) that maps each pair_id to {bmin, bmax}. Only ~4,096 unique pairs.

This creates a 2-cycle read pipeline (one cycle for the index lookup, one for the pair lookup), but it cuts BRAM usage roughly in half. The tradeoff is an extra clock cycle of latency per healthy range read, which the AF engine FSM accounts for with an additional wait state.

## 10. Real-Time Monitor (Deployment Mode)

After validation proved the FPGA works, I built a standalone deployment system. The FPGA-Real-Time-Monitor directory is self-contained it has its own copy of the Verilog modules, its own .mem files trained on all 452 users (not a 90/10 split), and its own Python CLI.

The setup is:
 - python fpga_realtime_monitor.py --setup trains on the full dataset and exports .mem files
 - Load those into Vivado, synthesize, program the board
 - python fpga_realtime_monitor.py --run --port COM3 opens an interactive session

From the CLI you can:
 - diagnose <user_id> send one patient from the dataset
 - diagnose 10 20 30 40 send 4 in parallel
 - diagnose all run all 452 patients in batches of 4
 - load patient_data.csv + diagnose me diagnose a new patient from CSV

This is the "product" mode. No golden model comparison, no validation metrics just send data, get a diagnosis. The idea is that you could plug an ECG sensor into this and have a bedside arrhythmia detector.

## 11. Power and Energy Measurement

The last thing I added was energy tracking using CodeCarbon. The fpga_uart_validator.py wraps Algorithm 4 inference (the part the FPGA replaces) in an EmissionsTracker. For each fold, it measures:
 - Total energy consumed (kWh)
 - Average power draw (watts)
 - CPU and RAM energy breakdown
 - Duration of inference

This gets saved per-fold in the emissions CSV files and aggregated in cv_summary.json. The point is to have a fair comparison: the FPGA running at 100 MHz and ~0.5W vs the laptop CPU running at 4 GHz and ~15-45W. At this scale with UART bottlenecking everything, the FPGA doesn't win on raw speed. But it wins massively on power which matters if you're putting this in a portable device.

## Summary of Roadblocks and Lessons

Throughout this project the biggest challenges were:

 1. Fixed-point precision choosing the right Q format for each variable required profiling every value across all 452 patients. Getting this wrong silently corrupts results. The reciprocal probability table (prob_pgt1) was the worst Q s2.14 would have clamped 78% of values, so I had to use Q s3.13 and accept slightly less fractional precision.

 2. BRAM fitting the Basys 3 has limited block RAM. The healthy range table almost didn't fit. Two-level lookup with pair deduplication was the solution.

 3. Integration timing every module has its own latency profile. BRAM reads are registered (1 cycle), the divider is pipelined (2 cycles), the multiplier uses a DSP slice. The top-level FSM has to account for all of these, and getting the handshakes wrong causes subtle data corruption that only shows up on certain patients.

 4. UART reliability serial communication at 115200 baud with 558 bytes per patient is slow and fragile. Synchronization between Python and the FPGA was a recurring source of bugs.

 5. Parallel architecture the jump from 1 lane to 4 lanes required rethinking every module's interface. But once you separate "what depends on the model" from "what depends on the patient", the partitioning becomes clear.








