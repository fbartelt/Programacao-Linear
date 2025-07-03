# Graph Coloring via Column Generation

This project estimates the chromatic number of undirected graphs using a column generation approach based on the formulation by Mehrotra and Trick, with cliques computed in the complement graph.

## Requirements

- Python version: 3.10
- CPLEX version: 22.1.1
- Python Packages:
  - `networkx`: for graph representation and manipulation

Install the required Python package with:

```bash

pip install networkx

```

Make sure CPLEX is correctly installed and accessible in your Python environment.

## File Structure
The following structure is expected for correct execution:

```kotlin

graph_coloring.py
data/
├── gcol1.txt
├── gcol2.txt
├── gcol3.txt
├── gcol4.txt
└── gcol5.txt

```

- `graph_coloring.py:` main script that runs the algorithm

- `data/:` folder containing input graph instances from the OR Library

## Running the Script

Run the main script from the root folder (same level as `data/`):

```bash
python graph_coloring.py
```

This will process all five instances (`gcol1.txt` through `gcol5.txt`) and output the lower bounds (fractional chromatic numbers) found by the linear relaxation, along with timing and iteration statistics.
