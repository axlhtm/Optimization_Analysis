# =============================================================================
# GENETIC ALGORITHM OPTIMIZATION - TRAVELING SALESMAN PROBLEM (TSP)
# =============================================================================

# Import Python Library 
import folium
import numpy as np
import os 
import pandas as pd 
import plotly.express as px
import pygad
import random

from geopy.distance import geodesic

# Problem Context 
'''
You're a delivery person for a mobile app that delivers coffee and pastries from various Starbucks locations.
You've received a surge of orders from customers in a specific area. To optimize delivery time and cost, 
you need to find the most efficient route to visit all these Starbucks locations, pick up the orders, and deliver them to the customers. 

Genetic Algorithm Simmulation Approach: 
    - This approach minimizes total delivery distance, saving time and fuel costs.
    - It ensures each Starbucks is visited exactly once, avoiding missed deliveries.
    - The visual representation on a map aids in understanding the delivery route.
'''

# Change Working Directories 
os.chdir("C:/Users/axelh/OneDrive/Documents/Hydroinformatics/Optimization_Analysis/")

# Import Datasets
data    = pd.read_csv('Starbucks_Route.csv')
df      = data[data['countryCode']=='GB']
df.reset_index(inplace=True)

# Group and Visualize Starbucks Branch by Town
vis = df.groupby('city').storeNumber.count().reset_index()
px.bar(vis, x='city', y='storeNumber', template='seaborn')
map = folium.Map(location=[51.509685, -0.118092], zoom_start=6, tiles="stamentoner")
for _, r in df.iterrows():
  folium.Marker(
      [r['latitude'], r['longitude']], popup=f'<i>{r["storeNumber"]}</i>'
  ).add_to(map)
map  

# Testing the Distance Methodology
origin  = (df['latitude'][0], df['longitude'][0])
dest    = (df['latitude'][100], df['longitude'][100])
geodesic(origin, dest).kilometers

# Preparing Data Structures
test    = df.head(10)
genes   = {store_num:[lat, lon] for store_num, lat, lon in zip(test['storeNumber'], test['latitude'], test['longitude'])}
stores  = list(genes.keys())
check_range = [i for i in range(0, 10)]

# Defining Objective Functions of GA
def build_population(size, chromosome_size):
  population = []
  for i in range(size):
    home_city   = 0
    added       = {home_city:'Added'}
    chromosome  = [home_city]
    while len(chromosome) < chromosome_size:
      proposed_gene = random.randint(0, chromosome_size-1)
      if added.get(proposed_gene) is None:
        chromosome.append(proposed_gene)
        added.update({proposed_gene:'Added'})
      else:
        pass
    chromosome.append(home_city)
    population.append(chromosome)
  return np.array(population)
population = build_population(100, 10)
population.shape

def fitness_func(solution, solution_idx, ga_instance):
    """
    Calculates the fitness of a given solution (chromosome).

    Args:
        solution: A list representing the order of Starbucks to visit.
        solution_idx: The index of the solution in the population (not currently used).
        ga_instance: The pygad.GA instance (not currently used).

    Returns:
        The fitness value (reciprocal of total distance) of the solution.
    """
    if not isinstance(solution, (list, np.ndarray)):
        raise TypeError(f"Expected solution to be a list or ndarray, got {type(solution)}")

    total_dist = 0
    for i in range(len(solution) - 1):  # Iterate up to second-to-last element
        a = genes.get(stores[solution[i]])
        b = genes.get(stores[solution[i + 1]])
        try:
            dist = geodesic(a, b).kilometers
        except IndexError:
            dist = 0
        total_dist += dist

    # Fitness is inversely proportional to distance (maximize fitness)
    fitness = 1 / total_dist
    return fitness



def pmx_crossover(parent1, parent2, sequence_start, sequence_end):
  # initialise a child
  child = np.zeros(parent1.shape[0])
  # get the genes for parent one that are passed on to child one
  parent1_to_child1_genes = parent1[sequence_start:sequence_end]
  # get the position of genes for each respective combination
  parent1_to_child1 =  np.isin(parent1,parent1_to_child1_genes).nonzero()[0]
  for gene in parent1_to_child1:
    child[gene] = parent1[gene]
  # gene of parent 2 not in the child
  genes_not_in_child = parent2[np.isin(parent2, parent1_to_child1_genes, invert=True).nonzero()[0]]  
  # if the gene is not already
  if genes_not_in_child.shape[0] >= 1:
    for gene in genes_not_in_child:
      if gene >= 1:
        lookup = gene
        not_in_sequence = True
        while not_in_sequence:
          position_in_parent2 = np.where(parent2==lookup)[0][0]
          if position_in_parent2 in range(sequence_start, sequence_end):
            lookup = parent1[position_in_parent2]
          else:
            child[position_in_parent2] = gene
            not_in_sequence = False
  return child

def crossover_func(parents, offspring_size, ga_instance):
  offspring = []
  idx = 0
  while len(offspring) != offspring_size[0]:
    # locate the parents
    parent1 = parents[idx % parents.shape[0], :].copy()
    parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
    # find gene sequence in parent 1 
    sequence_start = random.randint(1, parent1.shape[0]-4)
    sequence_end = random.randint(sequence_start, parent1.shape[0]-1)
    # perform crossover
    child1 = pmx_crossover(parent1, parent2, sequence_start, sequence_end)
    child2 = pmx_crossover(parent2, parent1, sequence_start, sequence_end)    
    offspring.append(child1)
    offspring.append(child2)
    idx += 1
  return np.array(offspring)

def mutation_func(offspring, ga_instance):
  for chromosome_idx in range(offspring.shape[0]):
    # define a sequence of genes to reverse
    sequence_start = random.randint(1, offspring[chromosome_idx].shape[0] - 2)
    sequence_end = random.randint(sequence_start, offspring[chromosome_idx].shape[0] - 1)
    genes = offspring[chromosome_idx, sequence_start:sequence_end]
    # start at the start of the sequence assigning the reverse sequence back to the chromosome
    index = 0
    if len(genes) > 0:
      for gene in range(sequence_start, sequence_end):
          offspring[chromosome_idx, gene] = genes[index]
          index += 1
    return offspring

def on_crossover(ga_instance, offspring_crossover):
    # apply mutation to ensure uniqueness 
    offspring_mutation  = mutation_func(offspring_crossover, ga_instance)
    # save the new offspring set as the parents of the next generation
    ga_instance.last_generation_offspring_mutation = offspring_mutation
    
def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)

# Initialising Genetic Algorithm
ga_instance = pygad.GA(num_generations      = 100,
                       num_parents_mating   = 40,
                       fitness_func         = fitness_func,
                       sol_per_pop          = 200,
                       initial_population   = population,
                       gene_space           = range(0, 10),
                       gene_type            = int,
                       mutation_type        = mutation_func,
                       on_generation        = on_generation,
                       crossover_type       = crossover_func, 
                       keep_parents         = 6,
                       mutation_probability = 0.4)
ga_instance.run()