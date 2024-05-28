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

Genetic Algorithm Simmulation Approach: 
    - By mimicking biological evolution, the algorithm iteratively improves solutions, 
      ultimately finding the optimal combination of items that maximizes the total value 
      within the capacity constraint.
    - The genetic algorithm provides an efficient way to explore a vast number of possible 
      item combinations and identify the most beneficial selection.
'''

# Change Working Directories 
os.chdir("C:/Users/axelh/OneDrive/Documents/Hydroinformatics/Optimization_Analysis/")

# Import Datasets
weights     = [3, 4, 5, 8, 10, 20, 2, 2, 3, 8]           # Weigth of each item (kg)
values      = [10, 40, 30, 15, 20, 5, 3, 15, 14, 2]      # Value of each item 
capacity    = 15                                         # Maximum capacity of the knapsack

# Check Total Weights 
total_weight = np.sum(weights)
if total_weight > capacity: 
    print("Total weight of all of your stuff is", total_weight, "kg. You need to select your items.")
else: 
    print("Total weight of all of your stuff is", total_weight, "kg. All items can be putted into your bag.")

# Number of items
n_items = len(values)

# Defining the Objective Functions of PSO
def objective_function(x):
    total_value = np.dot(x, values)
    total_weight = np.dot(x, weights)
    if total_weight > capacity:
        return -1  # Return a penalty if the weight exceeds the capacity
    return -total_value  # We return the negative value because pso minimizes the objective function

# Define the constraint function for the PSO
def constraint_function(x):
    return capacity - np.dot(x, weights)  # This should be >= 0

# Define the bounds for each item (0 or 1)
lb = [0] * n_items
ub = [1] * n_items

# Run PSO
xopt, fopt = pso(objective_function, lb, ub, swarmsize=100, maxiter=200, ieqcons=[constraint_function], debug=True)

# Convert the optimal continuous solution to binary (0 or 1)
xopt_binary = np.round(xopt)

print(f"Optimal item selection: {xopt_binary}")
print(f"Total value: {np.dot(xopt_binary, values)}")
print(f"Total weight: {np.dot(xopt_binary, weights)}")
