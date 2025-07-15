# TSP-Optimizer-Local-vs-Global-Search-with-Hill-Climbing-and-Simulated-Annealing

## Overview

This program solves the **Travelling Salesman Problem (TSP)** using two optimization techniques:

- **Hill Climbing** (Local Search)
- **Simulated Annealing** (Global Search)

It compares the quality of both solutions visually and numerically by computing the total tour distances.

---

## How It Works

1. The program loads TSP coordinates from a `.tsp` file (TSPLIB format).
2. It runs both optimization algorithms independently:
   - Hill Climbing tries to improve a random tour by local swaps.
   - Simulated Annealing probabilistically accepts worse solutions early on to escape local minima.
3. It calculates the total tour distance for both methods.
4. A side-by-side plot visualizes the two solutions.

---

## File Structure

- `main.py` – Main Python script.
- `ulysses16.tsp` – Sample TSP data file. You may replace this with another `.tsp` file in TSPLIB format from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

---

Install dependencies with:

```bash
pip install numpy matplotlib scipy
