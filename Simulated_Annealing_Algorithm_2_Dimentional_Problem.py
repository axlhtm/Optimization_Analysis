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
# EXAMPLE II: FIND THE MINIMUM VALUE OF A TWO-DIMENTIONAL PROBLEM
# =============================================================================

# Set the objective function for a two-dimentional problem
def objective_function_2D(x, y):
    """
    This function defines the objective function to be minimized.
    The X and Y value that will be chosen is derived from the simulated annealing function below.
    """
    # Example function with one variables
    return x**2 + y**2 - 6*x - 4*y + 13

# Set the simulated annealing function for a two-dimentional problem
def simulated_annealing_2D(objective, bounds, cooling_rate, n_iterations):
    """
    This function implements the simulated annealing algorithm.
    """
    ## Initial solution and evaluation
    best_solution       = np.array([random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1])])
    best_evaluation     = objective(best_solution[0], best_solution[1])
    current_solution    = best_solution.copy()
    current_evaluation  = best_evaluation
    temperature         = 1.0
    ## List to store all the solutions evaluated during the process
    all_solutions = []
    ## Run the algorithm for n_iterations
    for i in range(n_iterations):
        ## Generate a new candidate solution with small random steps for both x and y
        new_solution = current_solution + np.random.uniform(-1, 1, size=2) * 0.1
        # Clip the solution within bounds for both variables
        new_solution = np.clip(new_solution, bounds[0], bounds[1])
        ## Evaluate the candidate solution
        new_evaluation = objective(new_solution[0], new_solution[1])
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
          best_solution = current_solution.copy()
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
best_solution, best_evaluation, all_solutions = simulated_annealing_2D(objective_function_2D, bounds, cooling_rate, n_iterations)

# Plot the simulated annealing algorithm
def plot_2d_function():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function_2D(X, Y)
        ## Create a figure with a specific size and dpi
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ## Plot the objective function
    ax.contourf(X, Y, Z, cmap="viridis")
    # Plot the best solution
    ax.scatter(best_solution[0], best_solution[1], marker="o", color="white", label="Best Solution", s=100)
    # Plot all solutions visited
    for solution in all_solutions:
      ax.scatter(solution[0], solution[1], marker="x", color="blue", alpha=0.2)
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Simulated Annealing Optimization for a Two-DImentional Problem")
    ax.legend()
    plt.show()
plot_2d_function()