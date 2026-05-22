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

