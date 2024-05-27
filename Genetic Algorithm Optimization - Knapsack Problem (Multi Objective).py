# =============================================================================
# GENETIC ALGORITHM OPTIMIZATION - KNAPSACK PROBLEM 
# =============================================================================

# Import Python Library 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pygad

# Problem Context 
'''
Imagine you're a hiker preparing for a trek. You have various items (sleeping bag, tent, food, first-aid kit, etc.) 
with different weights, usefulness value, and fragility for the trip. Your backpack has a weight limit. 
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
weights     = np.array([3, 4, 5, 8, 10, 20, 2, 2, 3, 8])           # Weigth of each item (kg)
values      = np.array([10, 40, 30, 15, 20, 30, 10, 15, 10, 10])   # Value of each item 
fragility   = np.array([2, 1, 3, 4, 1, 2, 5, 1, 2, 5])             # Fragility of each item
capacity    = 15                                                   # Maximum bag capacity (kg)
risk        = 12                                                   # Maximum risk tolerance

# Create a Dataframe 
def dataframe():
    global df
    data = {
        "Weight"    : weights,
        "Value"     : values,
        "Fragility" : fragility
    }
    df = pd.DataFrame(data, index=["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6", "Item 7", "Item 8", "Item 9", "Item 10"])
dataframe()

# Defining the Objective Functions of GA
def fitness_func(ga_instance, solution, solution_idx):
    '''
    The multi objective functions for the knapsack problem will follow this rule: 
        1. Prioritize the capacity constraint.
        2. Maximize total value of items that can be brought.(Factor: 0.7)
        3. Minimize total fragility of items that can be brought. (Factor: 0.3)
    '''
    # Calculate selected item properties
    selected_weights = weights[solution == 1]
    selected_values = values[solution == 1]
    selected_fragility = fragility[solution == 1]
    # Check for capacity violation
    total_weight = np.sum(selected_weights)
    # Penalize solutions exceeding capacity (very high penalty for strict enforcement)
    if total_weight >= capacity:
      return [-1, -1]  # Return very low fitness for infeasible solutions
    # Normalize values and fragility (assuming 0 is the minimum value)
    normalized_values       = (selected_values - np.min(values)) / (np.max(values) - np.min(values))
    normalized_fragility    = (selected_fragility - np.min(fragility)) / (np.max(fragility) - np.min(fragility))
    # Weighted objective calculation
    weight_values            = 0.7  # Weight for maximizing total value
    weight_fragility        = 0.3  # Weight for minimizing total fragility
    # Combine objectives if capacity constraint is met
    fitness = [np.sum(normalized_values) * weight_values,  # Maximize total value (weight 1)
               np.sum(normalized_fragility) * weight_fragility,]  # Minimize total fragility (weight -1)
    return fitness

# Initialising Genetic Algorithm
ga_instance = pygad.GA(num_generations          = 30,               # Number of generations
                       num_parents_mating       = 2,                # Number of parents for crossover
                       fitness_func             = fitness_func,     # Our defined fitness function
                       sol_per_pop              = 10,               # Number of solutions in each population
                       num_genes                = len(weights),     # Number of genes (equal to item count)
                       gene_space               = [0, 1],           # Binary representation (0: exclude, 1: include)
                       mutation_by_replacement  = True,             # Mutation replaces existing gene
                       gene_type                = int,              # Gene type is integer
                       parent_selection_type    = "sss",            # Steady-state selection
                       keep_parents             = 1,                # Keep the best parent for next generation
                       crossover_type           = "single_point",   # Single-point crossover
                       mutation_type            = "random",         # Random mutation
                       mutation_num_genes       = 1,                # Mutate one gene at a time
                       initial_population       = np.random.randint(0, 2, size=(10, len(weights))),  # Random initial population
                       )

# Run the Genetic Algorithm
ga_instance.run()
solution_population = ga_instance.population
fitness_values = ga_instance.best_solutions_fitness

# Access Results 
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best Solution:", solution)
print("Number of items that can be bought is", sum(solution), "items.")
print("Total weight of the selected items:", np.sum(weights * solution), "kg.")
print("Total fragility risk:", np.sum(fragility * solution))
print("Total value of the selected items:", np.sum(values * solution)) 

selected_values = []
selected_fragilities = []
for solution in solution_population:
  selected_values.append(np.sum(values[solution == 1]))
  selected_fragilities.append(np.sum(fragility[solution == 1]))

# Extract value and fragility from fitness (assuming utility function)
fitness_values = np.array(fitness_values)
utility_values = fitness_values  # Ass

# Plot Pareto-optimal solutions
plt.scatter(selected_values, selected_fragilities)
plt.xlabel("Total Value")
plt.ylabel("Total Fragility")
plt.title("Pareto-Optimal Solutions")
plt.grid(True)
plt.show()

BEST_SOLUTION = ga_instance.best_solution()
