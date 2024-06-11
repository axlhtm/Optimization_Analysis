# =============================================================================
# SIMULATED ANNEALING ALGORITHM
# =============================================================================

'''
Simulated Annealing is inspired by the physical process of annealing, where metals are slowly cooled to 
reduce imperfections. There are several hyperparameters that can be tuned to affect the algorithm's performance:
    - Cooling Rate: 
        This value controls how quickly the temperature is reduced during the optimization process. 
        A higher cooling rate leads to faster cooling, potentially getting stuck in local minima. 
        A slower cooling rate allows for more exploration but might take longer to converge.
    - Number of Iterations: 
        This defines the number of times the algorithm iterates through the loop, generating and evaluating 
        candidate solutions. More iterations allow for more exploration but also take longer to run.
    - Bounds: 
        These define the search space for the variables (x and y in this case). 
        Choosing appropriate bounds can restrict the algorithm to a relevant region and improve efficiency.
        
Here's a breakdown of cooling rates:
    - High cooling rate (above 1): 
        Not possible. The cooling rate represents the factor by which the temperature is multiplied at 
        each iteration. A value greater than 1 would actually cause the temperature to increase, 
        which wouldn't be cooling at all.
    - Medium cooling rate (around 0.8 - 0.99): 
        This is a common range used in practice. It allows for a balance between exploration (finding new solutions) 
        and exploitation (converging towards the best solution found so far).
    - Low cooling rate (below 0.8): 
        This leads to slower cooling, allowing the algorithm to explore the search space more thoroughly. 
        This can be helpful for complex problems with many local minima, but it also increases the computation time.

Here's a breakdown of bounds:
    - For 1 variable (x):
        Define a single tuple with two elements representing the lower and upper bounds. 
        Eg: bounds = (-5, 5))
    - For 2 varialbes (x, y): 
        Define a list containing two tuples, each representing the bounds for a single variable. 
        Eg: bounds = [(-5, 5), (-2, 2)]
    - For 3 variables (x,y,z): 
        Define a list containing three tuples, each representing the bounds for a single variable.
        Eg: bounds = [(-5, 5), (-2, 2), (-1, 3)]
'''
# Import Python Libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# =============================================================================
# EXAMPLE I: FIND THE MINIMUM VALUE OF A ONE-DIMENTIONAL PROBLEM
# =============================================================================

# Set the objective function for a one-dimentional problem
def objective_function_1D(x):
  """
  This function defines the objective function to be minimized.
  The X value that will be chosen is derived from the simulated annealing function below.
  """
  # Example function with one variables
  return x**2 - 6*x + 9

# Set the simulated annealing function for a one-dimentional problem
def simulated_annealing_1D(objective, bounds, cooling_rate, n_iterations):
  """
  This function implements the simulated annealing algorithm.
  """
  ## Initial solution and evaluation
  best_solution         = random.uniform(bounds[0], bounds[1])
  best_evaluation       = objective(best_solution)
  current_solution      = best_solution
  current_evaluation    = best_evaluation
  temperature           = 1.0
  ## List to store all the solutions evaluated during the process
  all_solutions = []
  ## Run the algorithm for n_iterations
  for i in range(n_iterations):
    ## Generate a new candidate solution
    new_solution = current_solution + random.uniform(-1, 1) * 0.1  # Small random step
    ## Clip the solution within bounds
    new_solution = max(bounds[0], min(bounds[1], new_solution))
    ## Evaluate the candidate solution
    new_evaluation = objective(new_solution)
    ## Decide whether to accept the new solution
    delta_e = new_evaluation - current_evaluation
    if delta_e < 0:  # Always accept improvement
      current_solution = new_solution
      current_evaluation = new_evaluation
    else:
      p = random.uniform(0, 1)
      if p < np.exp(-delta_e / temperature):  # Metropolis acceptance criteria
        current_solution = new_solution
        current_evaluation = new_evaluation
    ## Update temperature
    temperature *= cooling_rate
    ## Update best solution if found a better one
    if current_evaluation < best_evaluation:
      best_solution = current_solution
      best_evaluation = current_evaluation
    ## Store all the solutions evaluated during each iteration
    all_solutions.append(current_solution)
  return best_solution, best_evaluation, all_solutions

# Set the hyperparameter of the simulated anealling algorithm
## Define the search bounds
bounds          = (-5, 5)
## Define the cooling rate
cooling_rate    = 0.85
## Define the number of iterations
n_iterations    = 1000

# Run the simulated anealling algorithm
best_solution, best_evaluation, all_solutions = simulated_annealing_1D(objective_function_1D, bounds, cooling_rate, n_iterations)

# Plot the simulated annealing algorithm
def plot_1D(): 
    ## Define figure size
    plt.figure(figsize=(8, 6), dpi=300)  
    ## Define the x and y axis
    x       = np.linspace(bounds[0], bounds[1], 400)
    y_func  = objective_function_1D(x)
    ## Plot the original function
    plt.plot(x, y_func, label='f(x) = x^2 - 6x + 9')
    ## Plot the evaluated points during the optimization process
    plt.scatter(all_solutions, [objective_function_1D(val) for val in all_solutions], label='Evaluated Solutions', c='red')
    ## Add labels and title
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Simulated Annealing Optimization for a One-DImentional Problem')
    ## Show the legend
    plt.legend()
    ## Show the plot with grid
    plt.grid(True)
    plt.show()
plot_1D()