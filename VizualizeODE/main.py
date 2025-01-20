import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def cpu_temperature_model(t, T, k, T_env):
    """ODE for CPU temperature dynamics based on Newton's Law of Cooling."""
    return -k * (T - T_env)

def main():
    """
    CPU Temperature Dynamics Solver
    This program solves and visualizes the cooling behavior of a CPU over time.
    Author: Brandon Leydon
    Date: 1/19/2025
    Dependencies: numpy, scipy, matplotlib
    """

    # User inputs
    try:
        T0 = float(input("Enter initial CPU temperature (in °C): "))
        T_env = float(input("Enter ambient temperature (in °C): "))
        k = float(input("Enter heat dissipation coefficient (in 1/min): "))
        t_end = float(input("Enter total simulation time (in minutes): "))
    except ValueError:
        print("Error: Please enter valid numeric values.")
        return

    if k <= 0:
        print("Error: Heat dissipation coefficient must be positive.")
        return

    # Time range for solution
    t_start = 0
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, 500)

    # Solve ODE
    solution = solve_ivp(cpu_temperature_model, t_span, [T0], args=(k, T_env), t_eval=t_eval)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(solution.t, solution.y[0], label="CPU Temperature (°C)", color='blue')
    plt.axhline(y=T_env, color='red', linestyle='--', label="Ambient Temperature (°C)")
    plt.title("CPU Temperature Dynamics Over Time", fontsize=16)
    plt.xlabel("Time (minutes)", fontsize=14)
    plt.ylabel("Temperature (°C)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    # Output
    print(f"\nSimulation Complete!")
    print(f"Initial CPU Temperature: {T0}°C")
    print(f"Ambient Temperature: {T_env}°C")
    print(f"Heat Dissipation Coefficient: {k} 1/min")
    print(f"Estimated final temperature after {t_end} minutes: {solution.y[0][-1]:.2f}°C")

if __name__ == "__main__":
    main()
