# Dynamic System Optimization

## Overview  
This repository showcases various **optimization techniques** applied to **dynamic systems**, developed within the scope of a university module.  
It contains clean, modular implementations of:

- **ARX Modeling** (AutoRegressive with eXogenous inputs)  
- **Linear Programming (LP)**  
- **Particle Swarm Optimization (PSO)**  
- **Genetic Algorithm (GA)**  

Each subproject is self-contained, with input/output examples and mathematical background where necessary.

## Key Objectives  
- Apply core optimization strategies to real-world dynamic system problems.  
- Explore both **deterministic** and **heuristic** methods.  
- Focus on **interpretable code**, **manual implementations** (no scikit-learn for modeling), and **scientific reproducibility**.

## Subprojects  

### 1. ARX Modeling  
- Implements a **manual ARX model** without external ML libraries.  
- Includes **data preprocessing**, **noise injection**, **signal smoothing**, and **model evaluation** (RMSE, MAE).  
- Input: CSV time series data (`u.csv`, `y.csv`).  
- Output: Error plots, prediction accuracy.

📁 [`/ARX-Model`](./ARX-Model)

---

### 2. Linear Programming  
- Solves **resource allocation and system constraint problems** using `scipy.optimize.linprog`.  
- Demonstrates **objective function formulation** and **constraint matrices**.

📁 [`/Linear-Programming`](./Linear-Programming)

---

### 3. Particle Swarm Optimization  
- Implements a **fully custom PSO algorithm** from scratch.  
- Used to **minimize benchmark functions** or user-defined loss functions.  
- Includes **3D visualization of convergence** and **parameter tuning interface**.

📁 [`/PSO`](./PSO)

---

### 4. Genetic Algorithm  
- Generic GA framework to solve **nonlinear optimization tasks**.  
- Modular design for crossover, mutation, selection strategies.  
- Tunable parameters: mutation rate, elitism, generation size.

📁 [`/Genetic-Algorithm`](./Genetic-Algorithm)

---

## Technologies Used  
- **Programming Language**: Python  
- **Libraries**: `numpy`, `matplotlib`, `scipy`, `pandas`  
- **Development Environment**: Jupyter Notebook, VS Code  

## Prerequisites  
Install the required packages:  
```bash
pip install numpy pandas matplotlib scipy sklearn


