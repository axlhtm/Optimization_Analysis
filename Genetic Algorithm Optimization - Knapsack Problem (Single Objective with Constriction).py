# =============================================================================
# GENETIC ALGORITHM OPTIMIZATION - KNAPSACK PROBLEM 
# =============================================================================

# Import Python Library 
import numpy as np
import os
import pandas as pd
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
item_name   = ["Item 0", "Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6", "Item 7", "Item 8", "Item 9"]
weights     = np.array([3, 4, 5, 8, 10, 20, 2, 2, 3, 8])           # Weigth of each item (kg)
values      = np.array([10, 40, 30, 15, 20, 5, 3, 15, 14, 2])      # Value of each item 
fragility   = np.array([2, 1, 3, 4, 1, 2, 5, 1, 2, 5])             # Fragility of each item
capacity    = 15                                                   # Maximum bag capacity (kg)
risk        = 12                                                   # Maximum risk tolerance

# Check Total Weights 
total_weight = np.sum(weights)
if total_weight > capacity: 
    print("Total weight of all of your stuff is", total_weight, "kg. You need to select your items.")
else: 
    print("Total weight of all of your stuff is", total_weight, "kg. All items can be putted into your bag.")

# Defining the Objective Functions of GA
def fitness_func(ga_instance, solution, solution_idx):
    '''
    The objective functions for the knapsack problem will follow this rule: 
        1. Item 2 and 10 must be brought
        2. The maximum weight value <= the capacity value
        3. The maximum fragility value <= the risk value
        4. Reach maximum value 
    '''
    ## Force inclusion of item 2 (index 1) by setting its solution value to 1
    solution[1] = 1
    ## Check for capacity condition
    if np.sum(weights * solution) <= capacity:
        ## Check for risk condition
        if np.sum(fragility * solution) <= risk: 
            ## Check for value condition
            return np.sum(values * solution)
        else: 
            return 0
    else: 
        return 0

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

# Access Results 
def results(): 
    global all_solution, best_solution
    solution_population = ga_instance.population
    ## Create a dataframe based on GA solutions
    all_solution                 = pd.DataFrame(solution_population, columns=item_name)
    all_solution['Total Weight'] = np.nan * np.ones(len(all_solution))
    all_solution['Total Values'] = np.nan * np.ones(len(all_solution))
    all_solution['Total Risk'] = np.nan * np.ones(len(all_solution))
    ## Calculate total weights and values
    for i in range (len(solution_population)): 
        all_solution.loc[[i], ['Total Weight']] = sum(solution_population[i] * weights)
        all_solution.loc[[i], ['Total Values']] = sum(solution_population[i] * values)
        all_solution.loc[[i], ['Total Risk']]   = sum(solution_population[i] * fragility)
    ## Slice the dataframe based on bag capacity
    all_solution    = all_solution[all_solution['Total Weight'] < capacity]
    best_solution   = all_solution.sort_values(by='Total Values', ascending=True).tail(1)
results()

solution, solution_fitness, solution_idx = ga_instance.best_solution()


