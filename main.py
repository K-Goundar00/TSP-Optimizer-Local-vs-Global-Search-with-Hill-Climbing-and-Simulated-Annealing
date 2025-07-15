import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import random
import os

# ============================================
# Function: Load TSP Data from File
# ============================================

def load_tsp_data(file_path):
    """
    Load coordinates from a TSP (TSPLIB format) file.
    Only reads lines between NODE_COORD_SECTION and EOF.
    """
    nodes = []
    reading_coords = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            if line.startswith("EOF"):
                break
            if reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        nodes.append((x, y))
                    except ValueError:
                        continue
    return np.array(nodes)

# ============================================
# Function: Generate a Random Tour
# ============================================

def generate_random_tour(n):
    """
    Generate a random permutation of n cities.
    """
    tour = list(range(n))
    random.shuffle(tour)
    return tour

# ============================================
# Function: Compute Total Distance of a Tour
# ============================================

def calculate_tour_distance(tour, coords):
    """
    Compute the total distance of a tour (cyclic path through all cities).
    """
    total = 0.0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        total += distance.euclidean(coords[tour[i]], coords[tour[j]])
    return total

# ============================================
# Hill Climbing TSP Solver
# ============================================

def hill_climbing_tsp(coords, max_iterations=10000):
    """
    Solve TSP using Hill Climbing.
    Start with a random tour and iteratively swap two cities to improve the path.
    """
    n = len(coords)
    current_tour = generate_random_tour(n)
    current_distance = calculate_tour_distance(current_tour, coords)

    for _ in range(max_iterations):
        i, j = random.sample(range(n), 2)
        neighbor_tour = current_tour.copy()
        neighbor_tour[i], neighbor_tour[j] = neighbor_tour[j], neighbor_tour[i]
        neighbor_distance = calculate_tour_distance(neighbor_tour, coords)

        if neighbor_distance < current_distance:
            current_tour = neighbor_tour
            current_distance = neighbor_distance

    return current_tour, current_distance

# ============================================
# Simulated Annealing TSP Solver
# ============================================

def simulated_annealing_tsp(coords, initial_temp=10000, cooling_rate=0.003, max_iterations=10000):
    """
    Solve TSP using Simulated Annealing.
    Probabilistically accept worse solutions early to escape local optima.
    """
    n = len(coords)
    current_tour = generate_random_tour(n)
    current_distance = calculate_tour_distance(current_tour, coords)

    best_tour = current_tour.copy()
    best_distance = current_distance
    temp = initial_temp

    for _ in range(max_iterations):
        i, j = random.sample(range(n), 2)
        neighbor_tour = current_tour.copy()
        neighbor_tour[i], neighbor_tour[j] = neighbor_tour[j], neighbor_tour[i]
        neighbor_distance = calculate_tour_distance(neighbor_tour, coords)

        delta = neighbor_distance - current_distance
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_tour = neighbor_tour
            current_distance = neighbor_distance

            if current_distance < best_distance:
                best_tour = current_tour.copy()
                best_distance = current_distance

        temp *= 1 - cooling_rate

    return best_tour, best_distance

# ============================================
# Plotting Function for Solution Comparison
# ============================================

def plot_tsp_solutions(coords, hc_tour, hc_distance, sa_tour, sa_distance):
    """
    Display a side-by-side plot of Hill Climbing vs. Simulated Annealing solutions.
    """
    plt.figure(figsize=(12, 6))

    # Hill Climbing
    plt.subplot(1, 2, 1)
    hc_coords = coords[np.array(hc_tour + [hc_tour[0]])]
    plt.plot(hc_coords[:, 0], hc_coords[:, 1], 'b-')
    plt.scatter(coords[:, 0], coords[:, 1], c='red')
    plt.title(f"Hill Climbing\nDistance: {hc_distance:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # Simulated Annealing
    plt.subplot(1, 2, 2)
    sa_coords = coords[np.array(sa_tour + [sa_tour[0]])]
    plt.plot(sa_coords[:, 0], sa_coords[:, 1], 'b-')
    plt.scatter(coords[:, 0], coords[:, 1], c='red')
    plt.title(f"Simulated Annealing\nDistance: {sa_distance:.2f}")
    plt.xlabel("X Coordinate")

    plt.tight_layout()
    plt.show()

# ============================================
# Main Execution
# ============================================

def main():
    # Change this to use a different TSP file placed in the same directory
    tsp_file = "ulysses16.tsp"

    if not os.path.exists(tsp_file):
        print(f"Error: TSP file '{tsp_file}' not found in the directory.")
        return

    coords = load_tsp_data(tsp_file)
    if len(coords) == 0:
        print("Error: No valid coordinates found in the TSP file.")
        return

    print(f"Loaded TSP file: {tsp_file} with {len(coords)} cities.")

    # Solve using Hill Climbing
    print("Solving with Hill Climbing...")
    hc_tour, hc_distance = hill_climbing_tsp(coords)
    print(f"Hill Climbing Tour Distance: {hc_distance:.2f}")

    # Solve using Simulated Annealing
    print("Solving with Simulated Annealing...")
    sa_tour, sa_distance = simulated_annealing_tsp(coords)
    print(f"Simulated Annealing Tour Distance: {sa_distance:.2f}")

    # Display the results
    plot_tsp_solutions(coords, hc_tour, hc_distance, sa_tour, sa_distance)

if __name__ == "__main__":
    main()
