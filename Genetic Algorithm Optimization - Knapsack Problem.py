# =============================================================================
# GENETIC ALGORITHM OPTIMIZATION - KNAPSACK PROBLEM 
# =============================================================================

# Import Python Library 
import numpy as np
import os
import pygad

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
weights     = np.array([3, 4, 5, 8, 10, 20, 2, 2, 3])           # Weigth of each item (kg)
values      = np.array([10, 40, 30, 15, 20, 30, 10, 5, 10])     # Value of each item 
capacity    = 15                                                # Maximum bag capacity (kg)

# Check Total Weights 
total_weight = np.sum(weights)
if total_weight > capacity: 
    print("Total weight of all of your stuff is", total_weight, "kg. You need to select your items.")
else: 
    print("Total weight of all of your stuff is", total_weight, "kg. All items can be putted into your bag.")

# Defining the Objective Functions of GA
def fitness_func(ga_instance, solution, solution_idx):
    # Check for capacity violation
    if np.sum(weights * solution) > capacity:
        return 0  # Penalty for exceeding capacity
    # Calculate total value
    return np.sum(values * solution)

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

# Access Results 
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best Solution:", solution)
print("Number of items that can be bought is", sum(solution), " items.")
print("Total Weight:", np.sum(weights * solution))
print("Total Value:", np.sum(values * solution))