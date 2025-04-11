# Pathfinding Visualizer

A PyGame-based interactive visualization tool that demonstrates multiple pathfinding algorithms on a customizable grid.

## Description

This Pathfinding Visualizer is an educational tool that lets you see how different pathfinding algorithms work in real-time. Place start and end points, build walls, and watch algorithms like A*, Dijkstra's, and more find their way through the maze.

## Features

- **Multiple Algorithms**: Visualize 10 different pathfinding algorithms:
  - Bidirectional A*
  - Dijkstra's Algorithm
  - A* Search
  - Depth-First Search (DFS)
  - Greedy Best-First Search
  - Theta*
  - Bidirectional Search
  - Flow Field
  - Jump Point Search (JPS)
  - Hierarchical Pathfinding A* (HPA*)

- **Interactive Grid**: Create and modify mazes with mouse clicks
- **Customizable Environment**: Randomly generated obstacles
- **Real-time Visualization**: Watch each step of the algorithm's execution
- **Pause/Resume**: Control the visualization with spacebar
- **Clear Interface**: Color-coded nodes and helpful legend

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pathfinding-visualizer.git

# Navigate to the directory
cd pathfinding-visualizer

# Install dependencies
pip install pygame

# Run the program
python pathfinding_visualizer.py
```

## Usage

1. Run the program
2. Left-click to place start node (green)
3. Left-click to place end node (red)
4. Left-click to draw walls/obstacles (black)
5. Right-click to erase nodes
6. Press keys 0-9 to run different algorithms
7. Press SPACE to pause/resume visualization
8. Press ESC or C to stop the algorithm

## Controls

- **0**: Bidirectional A*
- **1**: Dijkstra's Algorithm
- **2**: A* Search
- **3**: Depth-First Search
- **4**: Greedy Best-First Search
- **5**: Theta*
- **6**: Bidirectional Search
- **7**: Flow Field
- **8**: Jump Point Search
- **9**: HPA*
- **ESC/C**: Stop Algorithm
- **SPACE**: Pause/Resume
- **Left Click**: Place Start/End/Wall
- **Right Click**: Erase
