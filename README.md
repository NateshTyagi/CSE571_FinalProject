# CSE 571 : Artificial Intelligence
## Final Project Topic 1 : Bi-directional search in Pac-Man Domain

We implement Bi-directional search for path finding problems using MM and MM0 algorithm in the Pac-Man domain as described in the paper titled **"Bi-directional Search That Is Guaranteed to Meet in the Middle",** _Robert C. Holte, Ariel Felner, Guni Sharon, Nathan R. Sturtevant, AAAI 2016_.

We have done a comparative study between BFS, DFS, A*, UCS, MM and MM0 search algorithms. We tested the algorithms in different Pac-Man maze layouts which ranged in size, complexity of the layout (the number of walls) and different start-goal position pairs.

## Dependencies
    Python 3.6

## How to run
* Open a terminal
* Clone this repository using : 

        git clone https://github.com/NateshTyagi/CSE571_FinalProject.git
* Navigate to the folder : _CSE571_FinalProject_
* Run the executable file on your terminal using the command: 

        ./complexityTest.sh  Argument1 Argument2 Argument3
  - Argument1(maze) : {tiny, medium, big}
  - Argument2(complexity) : {0, 30, 50}
  - Argument3(algorithm): {bfs, dfs, ucs, astar, MM, MM0}

* Addtionally, to run on smallMaze, contoursMaze, openMaze:

        python3 pacman.py -l {Argument1} -z .5 -p SearchAgent -a fn={Argument2},heuristic=manhattanHeuristic
  - Argument1(maze) : {smallMaze, contoursMaze, openMaze}
  - Argument2(algorithm): {bfs, dfs, ucs, astar, MM, MM0}
