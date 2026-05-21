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
 


