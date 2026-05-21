# PROJECT TIMELINE & OBJECTIVES

**Short Disclaimer**
This .md file is for people that want to understand what process as well as stuggles I went through to complete this project. Through this approach I will learn a lot and hit many roadblocks, which I hope to overcome. By following this file, you can see what my objectives, by path to implementation, and learning is.


## 1. Fixed-Point Analysis and Model Export
 a. Objective: Convert all floating-point computations in Algorithm 4 to fixed-point arithmetic and export trained model parameters to a format loadable by the FPGA.
 - Why: FPGAs don't have native floating-point units. Every multiplication, division, and comparison in Algorithm 4 must be expressed in fixed-point (e.g., Q16.16 or Q8.24 format). Choosing the wrong format causes either overflow (too few integer bits) or loss of precision (too few fractional bits) -- both silently corrupt results.

 First we must profile all the variables in Algorithm 4. This is the algorithm that will be implemented in FPGA architecture. Therefore we must know the range of the variables used here.