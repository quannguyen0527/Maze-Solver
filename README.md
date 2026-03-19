# Silent Cartographer: Maze Navigation Project

**Course:** COSC 4368 AI – Spring 2026  
**Instructor:** Dr. Raunak Sarbajna  
**TA:** Mert Saritac  
**Team:**   

---

## 📖 Project Overview

This project implements an intelligent agent capable of navigating 64x64 mazes with and without hazards.  
For the second checkpoint, we demonstrate:

1. Loading the maze into Python.  
2. Solving the maze (without hazards) using the A* algorithm.  
3. Visualizing the solution path.  
4. Loading hazards (death pits, teleport pads, and confusion traps) and demonstrating their behavior.  
5. Running a naive agent to show basic interaction with the environment.

---

## 🗂 Project Structure

maze-project/
│
├── main.py # Entry point: runs demos and naive agent
├── agent.py # NaiveAgent implementation
├── maze_loader.py # Loads PNG mazes into grids
├── planner.py # A* pathfinding implementation
├── visualizer.py # Visualization of maze + path
├── hazard_loader.py # Hazard detection from PNG
├── environment.py # Provided environment by instructor
│
└── mazes/
├── maze1.png # Maze without hazards
├── maze2.png # Maze with hazards


---

## 🛠 Requirements

Python 3.8+  

Required packages:

```bash
pip install pillow numpy matplotlib
🚀 How to Run
Make sure you are in the project folder:

cd path/to/maze-project
Run the main file:

python main.py
The program will:

Load and print maze grid

Solve maze using A*

Visualize solution path (pop-up window)

Load hazards and demonstrate their effects

Run a naive agent to show basic environment interaction

🧩 Features
Maze Loader: Converts PNG maze images to numeric grid representation.

A Pathfinding:* Finds shortest path from start to goal.

Visualization: Displays maze and solution path using matplotlib.

Hazard Loader: Detects pits, teleport pads, and confusion traps.

Naive Agent: Random movement agent demonstrating environment interaction.

📚 AI Assistance
We used ChatGPT (GPT-5.3) for:

Structuring project files

Implementing A* algorithm

Debugging PNG loader and visualization code

Prompts used:

“Give me an A* implementation for a grid maze in Python”

“Load PNG maze into numpy array”

“Visualize a path in a 2D numpy grid using matplotlib”

All code was reviewed and adapted to meet project specifications.

📈 Future Work
Implement hazard-aware agent

Improve exploration efficiency

Animate agent movement for demo

Optimize pathfinding for large, hazard-filled mazes

📄 License / Attribution
Environment and API provided by course instructors.

Code partially assisted by ChatGPT (GPT-5.3).

Any third-party libraries used (Pillow, NumPy, Matplotlib) are open-source.