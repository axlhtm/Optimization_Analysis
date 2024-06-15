# =============================================================================
# NUMERICAL METHOD - ITERATIVE APPROXIMATION
# =============================================================================
'''
Context:
You have a furnace that heats the house, but it can't maintain a perfectly constant temperature.
Outside temperature also affects the inside temperature.
You want to develop a model that predicts the inside temperature based on the furnace setting (power level) 
and the outside temperature.
'''


def adjust_furnace(desired_temp, outside_temp, current_temp):
  """
  This function iteratively adjusts the furnace setting to reach the desired temperature.

  Args:
      desired_temp: The desired inside temperature.
      outside_temp: The outside temperature.
      current_temp: The current inside temperature.

  Returns:
      The adjusted furnace setting (power level).
  """
  # This is a simplified model, and the actual relationship might be more complex.
  # Here, we assume the furnace increases the temperature by a fixed amount per unit power.
  heating_rate = 1  # Adjust this value based on your furnace and house characteristics
  # Initial guess for furnace setting (assuming some power is needed)
  furnace_setting = 0.5

  for i in range(10):  # Maximum of 10 adjustments
    # Predict the resulting temperature based on the current setting
    predicted_temp = current_temp + heating_rate * furnace_setting - outside_temp

    # Check if we reached the desired temperature within an acceptable error range (e.g., 1 degree) 
    error = abs(desired_temp - predicted_temp)
    if error < 1:
      return furnace_setting

    # Adjust the furnace setting based on the error
    if predicted_temp > desired_temp:
      # Too hot, decrease setting
      furnace_setting -= 0.1
    else:
      # Too cold, increase setting
      furnace_setting += 0.1

  print("Couldn't reach desired temperature within", 10, "iterations.")
  return furnace_setting

# Example usage:
desired_temp = 20  # Celsius
outside_temp = 10  # Celsius
current_temp = 15  # Celsius

furnace_setting = adjust_furnace(desired_temp, outside_temp, current_temp)
print("Furnace setting:", furnace_setting)
