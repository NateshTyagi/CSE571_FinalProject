#!/bin/bash


python pacman.py -l "$1"Maze"$2" -z .5 -p SearchAgent -a fn="$3",heuristic=manhattanHeuristic
