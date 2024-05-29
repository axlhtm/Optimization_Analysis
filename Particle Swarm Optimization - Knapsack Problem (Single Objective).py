# =============================================================================
# PARTICLE SWARM OPTIMIZATION - KNAPSACK PROBLEM 
# =============================================================================

# Import Python Library 
import numpy as np
import os

from pyswarm import pso

# Problem Context 
'''
Imagine you're a hiker preparing for a trek. You have various items (sleeping bag, tent, food, first-aid kit, etc.) 
with different weights and usefulness (value) for the trip. Your backpack has a weight limit. 
The knapsack problem asks you to choose the best combination of items that maximizes the overall 
usefulness (value) for your trek while staying within the weight limit of your backpack.

This code utilizes PSO, a population-based optimization technique inspired by the behavior of swarming birds.
Here's how it works:
    - Particles and Bounds: A swarm of particles is created, each representing a potential solution. 
      Each particle has a position (item selection) and a velocity. The position is a set of values between 0 and 1 
      (defined by lb and ub). A value closer to 0 indicates less of the item, and a value closer to 1 indicates more.
    - Result Interpretation: The final solution (xopt_binary) is obtained by rounding the continuous values to 0 or 1.
      This indicates which items are actually included (1) or excluded (0) in the optimal selection for your trek.
'''

# Change Working Directories 
os.chdir("C:/Users/axelh/OneDrive/Documents/Hydroinformatics/Optimization_Analysis/")

# Import Datasets
item_name   = ["Item 0", "Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6", "Item 7", "Item 8", "Item 9"]
weights     = [3, 4, 5, 8, 10, 20, 2, 2, 3, 8]           # Weigth of each item (kg)
values      = [10, 40, 30, 15, 20, 5, 3, 15, 14, 2]      # Value of each item 
capacity    = 15                                         # Maximum capacity of the knapsack

# Check Total Weights 
total_weight = np.sum(weights)
if total_weight > capacity: 
    print("Total weight of all of your stuff is", total_weight, "kg. You need to select your items.")
else: 
    print("Total weight of all of your stuff is", total_weight, "kg. All items can be putted into your bag.")

# Defining the Objective Functions of the Particle Swarm Optimization
def objective_function(x):
    total_value = np.dot(x, values)
    total_weight = np.dot(x, weights)
    if total_weight > capacity:
        return -1  
    return -total_value

# Define the Constraint Function for the Particle Swarm Optimization
def constraint_function(x):
    return capacity - np.dot(x, weights)

# Initialising Particle Swarm Optimization
## Defining lower and upper bound of PSO
lb  = [0] * len(values) 
ub  = [1] * len(values) 
## Defining number of itteration of PSO
n   = 100
## Initialize list to store positions of PSO
swarm_positions = []
swarm_binary    = []

# Run Particle Swarm Optimization
for _ in range(n):
  xopt, fopt = pso(objective_function,
                    lb,
                    ub,
                    swarmsize   = 100,
                    maxiter     = n,
                    ieqcons     = [constraint_function],
                    debug       = True)
  swarm_positions.append(xopt)
  swarm_binary.append(np.round(xopt)) 

# Access multiple solutions (rounded)
#for i, positions in enumerate(swarm_positions):
#  solution_binary = np.round(positions)
#  print(f"Solution {i+1} item selection: {solution_binary}")
#  print(f"Solution {i+1} total value: {np.dot(solution_binary, values)}")
#  print(f"Solution {i+1} total weight: {np.dot(solution_binary, weights)}")
#  print("-"*30)
  
# Access Results 
#xopt_binary = np.round(xopt)

#print(f"Optimal item selection: {xopt_binary}")
#print(f"Total value: {np.dot(xopt_binary, values)}")
#print(f"Total weight: {np.dot(xopt_binary, weights)}")
