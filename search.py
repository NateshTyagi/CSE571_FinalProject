# # search.py
# # ---------
# # Licensing Information:  You are free to use or extend these projects for
# # educational purposes provided that (1) you do not distribute or publish
# # solutions, (2) you retain this notice, and (3) you provide clear
# # attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# #
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# # The core projects and autograders were primarily created by John DeNero
# # (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# # Student side autograding was added by Brad Miller, Nick Hay, and
# # Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
#
# """
# In search.py, you will implement generic search algorithms which are called by
# Pacman agents (in searchAgents.py).
# """
#
# import util
#
#
# class SearchProblem:
#     """
#     This class outlines the structure of a search problem, but doesn't implement
#     any of the methods (in object-oriented terminology: an abstract class).
#
#     You do not need to change anything in this class, ever.
#     """
#
#     def getStartState(self):
#         """
#         Returns the start state for the search problem.
#         """
#         util.raiseNotDefined()
#
#     def isGoalState(self, state):
#         """
#           state: Search state
#
#         Returns True if and only if the state is a valid goal state.
#         """
#         util.raiseNotDefined()
#
#     def getSuccessors(self, state):
#         """
#           state: Search state
#
#         For a given state, this should return a list of triples, (successor,
#         action, stepCost), where 'successor' is a successor to the current
#         state, 'action' is the action required to get there, and 'stepCost' is
#         the incremental cost of expanding to that successor.
#         """
#         util.raiseNotDefined()
#
#     def getCostOfActions(self, actions):
#         """
#          actions: A list of actions to take
#
#         This method returns the total cost of a particular sequence of actions.
#         The sequence must be composed of legal moves.
#         """
#         util.raiseNotDefined()
#
#
# def tinyMazeSearch(problem):
#     """
#     Returns a sequence of moves that solves tinyMaze.  For any other maze, the
#     sequence of moves will be incorrect, so only use this for tinyMaze.
#     """
#     from game import Directions
#     s = Directions.SOUTH
#     w = Directions.WEST
#     return [s, s, w, s, w, w, s, w]
#
#
# def depthFirstSearch(problem):
#     """
#     Search the deepest nodes in the search tree first.
#
#     Your search algorithm needs to return a list of actions that reaches the
#     goal. Make sure to implement a graph search algorithm.
#
#     To get started, you might want to try some of these simple commands to
#     understand the search problem that is being passed in:
#
#     print("Start:", problem.getStartState())
#     print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
#     print("Start's successors:", problem.getSuccessors(problem.getStartState()))
#     """
#     "*** YOUR CODE HERE ***"
#     closed_set = set()
#     fringe = util.Stack()
#     path_stack = util.Stack()
#     path_from_root = []
#     fringe.push([problem.getStartState(), 'INITIAL', 0])
#     while True:
#         top_node = fringe.pop()
#         if top_node is None: return path_from_root
#         closed_set.add(top_node[0])
#         if problem.isGoalState(top_node[0]): return path_from_root
#         for successor in problem.getSuccessors(top_node[0]):
#             if successor[0] not in closed_set:
#                 fringe.push(successor)
#                 path_stack.push(path_from_root + [successor[1]])
#         path_from_root = path_stack.pop()
#
#
# def breadthFirstSearch(problem):
#     """Search the shallowest nodes in the search tree first."""
#     "*** YOUR CODE HERE ***"
#     closed_set = []
#     fringe = util.Queue()
#     path_queue = util.Queue()
#     path_from_root = []
#     print("Start State :", problem.getStartState())
#     fringe.push(problem.getStartState())
#     while True:
#         if not fringe.isEmpty():
#             front_node = fringe.pop()
#         else:
#             return path_from_root
#         if problem.isGoalState(front_node): return path_from_root
#         if front_node not in closed_set:
#             closed_set.append(front_node)
#             for successor in problem.getSuccessors(front_node):
#                 if successor[0] not in closed_set:
#                     fringe.push(successor[0])
#                     path_queue.push(path_from_root + [successor[1]])
#         if not path_queue.isEmpty():
#             path_from_root = path_queue.pop()
#
#
# def uniformCostSearch(problem):
#     """Search the node of least total cost first."""
#     "*** YOUR CODE HERE ***"
#     closed_set = set()
#     fringe = util.PriorityQueue()
#     path_queue = util.PriorityQueue()
#     path_from_root = []
#     fringe.push((problem.getStartState(), 0), 0)
#     while True:
#         front_node = fringe.pop()
#         if front_node[0] is None: return path_from_root
#         if problem.isGoalState(front_node[0]): return path_from_root
#         if front_node[0] not in closed_set:
#             closed_set.add(front_node[0])
#             for successor in problem.getSuccessors(front_node[0]):
#                 if successor[0] not in closed_set:
#                     fringe.push((successor[0], front_node[1] + successor[2]), front_node[1] + successor[2])
#                     path_queue.push(path_from_root + [successor[1]], front_node[1] + successor[2])
#         path_from_root = path_queue.pop()
#
#
# def nullHeuristic(state, problem=None):
#     """
#     A heuristic function estimates the cost from the current state to the nearest
#     goal in the provided SearchProblem.  This heuristic is trivial.
#     """
#     return 0
#
#
# def aStarSearch(problem, heuristic=nullHeuristic):
#     """Search the node that has the lowest combined cost and heuristic first."""
#     "*** YOUR CODE HERE ***"
#     closed_set = []
#     fringe = util.PriorityQueue()
#     path_queue = util.PriorityQueue()
#     path_from_root = []
#     fringe.push((problem.getStartState(), 0), 0)
#     while True:
#         front_node = fringe.pop()
#         if front_node[0] is None: return path_from_root
#         if problem.isGoalState(front_node[0]): return path_from_root
#         if front_node[0] not in closed_set:
#             closed_set.append(front_node[0])
#             for successor in problem.getSuccessors(front_node[0]):
#                 if successor[0] not in closed_set:
#                     f = front_node[1] + successor[2] + heuristic(successor[0], problem)
#                     fringe.push((successor[0], front_node[1] + successor[2]), f)
#                     path_queue.push(path_from_root + [successor[1]], f)
#         path_from_root = path_queue.pop()
#
#
# direction = {'North': 'South', 'East': 'West', 'South': 'North', 'West': 'East'}
#
#
# def bDSearchMM0(problem):
#     # mm0 is essentially bidrectional BFS
#     # initialize everything we need, so 2 queues, and 2 visited position dicts
#     q1, q2 = util.Queue(), util.Queue()
#
#     # Note these are dicts because each key is a visited position
#     # the value for each of those keys is the path we took to get there
#     visitedPos1, visitedPos2 = {}, {}
#
#     # Set up initial starting conditions
#     # so q1 starts from start and q2 starts from goal
#     q1.push(problem.getStartState())
#     q2.push(problem.goal)
#
#     # Mark those states as visited
#     visitedPos1[problem.getStartState()] = ''
#     visitedPos2[problem.goal] = ''
#
#     # Run until either queue is empty meaning there is no path from start to goal
#     while not q1.isEmpty() and not q2.isEmpty():
#         # While q2 is not empty
#         while not q1.isEmpty():
#             # Pop current pos and current path to get to that ops off priority queue
#             cpos1 = q1.pop()
#
#             # Check if we have a goal state
#             if problem.isGoalState(cpos1, visitedPos2):
#                 revd = [direction[x] for x in visitedPos2[cpos1]]
#                 # If so, reverse the other path that other search took to meet us, then append
#                 return visitedPos1[cpos1] + revd[::-1]
#
#             # If no goal state, expand all successors
#             for state in problem.getSuccessors(
#                     cpos1):  # Priority queue manages order for us so we don't have to use if statements
#                 if state[0] in visitedPos1:  # If already visited, don't visit again
#                     continue
#
#                 # Push each state and mark visited
#                 q1.push(state[0])
#                 visitedPos1[state[0]] = list(visitedPos1[cpos1]) + [state[1]]
#
#         # Second bfs instance, same as first just searching in reverse!
#         while not q2.isEmpty():
#             cpos2 = q2.pop()
#
#             if problem.isGoalState(cpos2, visitedPos1):
#                 return [direction[x] for x in visitedPos1[cpos2]][::-1] + visitedPos2[cpos2]
#
#             for state in problem.getSuccessors(cpos2):
#                 if state[0] in visitedPos2:
#                     continue
#
#                 q2.push(state[0])
#                 visitedPos2[state[0]] = list(visitedPos2[cpos2]) + [state[1]]
#     return []
#
#
# def bDSearchMM(problem, heuristic):
#     # Now MM search, which is astar in each direction
#     q1, q2 = util.PriorityQueue(), util.PriorityQueue()
#
#     # Declare dict to store visited positions
#     visitedPos1, visitedPos2 = {}, {}
#
#     # Add both starting states to visited Dicts
#     visitedPos1[problem.getStartState()] = []
#     visitedPos2[problem.goal] = []
#
#     # We use a priority que to store nodes in the frontier with the A* cost metric of f(n)=h(n)+g(n)
#     # problem.getCostOfActions() = g(n)
#     # heuristic(state, problem) = f(n)
#     q1.push((problem.getStartState()),
#             (problem.getCostOfActions({}) + heuristic(problem.getStartState(), problem, "g")))
#     q2.push((problem.goal), (problem.getCostOfActions({}) + heuristic(problem.goal, problem, "s")))
#
#     # Run while both frontier's are not empty and return [] in the case the goal is not reachable from the start
#     while not q1.isEmpty() and not q2.isEmpty():
#
#         # Run both searches at simultaneously
#         cpos1 = q1.pop()
#
#         if problem.isGoalState(cpos1, visitedPos2):
#             revd = [direction[x] for x in visitedPos2[cpos1]]
#             return visitedPos1[cpos1] + revd[::-1]
#
#         successors = problem.getSuccessors(cpos1)
#
#         for state in successors:  # priority queue manages order for us so we don't have to use if statements
#             if state[0] in visitedPos1:
#                 continue
#
#             visitedPos1[state[0]] = list(visitedPos1[cpos1]) + [state[1]]
#             q1.push(state[0], (problem.getCostOfActions(visitedPos1[state[0]]) + heuristic(state[0], problem, "g")))
#
#         cpos2 = q2.pop()
#
#         if problem.isGoalState(cpos2, visitedPos1):
#             return visitedPos1[cpos2] + [direction[x] for x in visitedPos2[cpos2]][::-1]
#
#         successors = problem.getSuccessors(cpos2)
#
#         for state in successors:  # priority queue manages order for us so we don't have to use if statements
#             if state[0] in visitedPos2:
#                 continue
#
#             visitedPos2[state[0]] = list(visitedPos2[cpos2]) + [state[1]]
#             q2.push(state[0],
#                     (problem.getCostOfActions(visitedPos2[state[0]]) + heuristic(state[0], problem, "s")))
#
#     return []
#
#
# # Abbreviations
# bfs = breadthFirstSearch
# dfs = depthFirstSearch
# astar = aStarSearch
# ucs = uniformCostSearch
# bd0 = bDSearchMM0
# bd = bDSearchMM

# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    closed_set = set()
    fringe = util.Stack()
    path_stack = util.Stack()
    path_from_root = []
    fringe.push([problem.getStartState(), 'INITIAL', 0])
    while True:
        top_node = fringe.pop()
        if top_node is None: return path_from_root
        closed_set.add(top_node[0])
        if problem.isGoalState(top_node[0]): return path_from_root
        for successor in problem.getSuccessors(top_node[0]):
            if successor[0] not in closed_set:
                fringe.push(successor)
                path_stack.push(path_from_root + [successor[1]])
        path_from_root = path_stack.pop()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed_set = []
    fringe = util.Queue()
    path_queue = util.Queue()
    path_from_root = []
    print("Start State :", problem.getStartState())
    fringe.push(problem.getStartState())
    while True:
        if not fringe.isEmpty():
            front_node = fringe.pop()
        else:
            return path_from_root
        if problem.isGoalState(front_node): return path_from_root
        if front_node not in closed_set:
            closed_set.append(front_node)
            for successor in problem.getSuccessors(front_node):
                if successor[0] not in closed_set:
                    fringe.push(successor[0])
                    path_queue.push(path_from_root + [successor[1]])
        if not path_queue.isEmpty():
            path_from_root = path_queue.pop()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    closed_set = set()
    fringe = util.PriorityQueue()
    path_queue = util.PriorityQueue()
    path_from_root = []
    fringe.push((problem.getStartState(), 0), 0)
    while True:
        front_node = fringe.pop()
        if front_node[0] is None: return path_from_root
        if problem.isGoalState(front_node[0]): return path_from_root
        if front_node[0] not in closed_set:
            closed_set.add(front_node[0])
            for successor in problem.getSuccessors(front_node[0]):
                if successor[0] not in closed_set:
                    fringe.push((successor[0], front_node[1] + successor[2]), front_node[1] + successor[2])
                    path_queue.push(path_from_root + [successor[1]], front_node[1] + successor[2])
        path_from_root = path_queue.pop()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    closed_set = []
    fringe = util.PriorityQueue()
    path_queue = util.PriorityQueue()
    path_from_root = []
    fringe.push((problem.getStartState(), 0), 0)
    while True:
        front_node = fringe.pop()
        if front_node[0] is None: return path_from_root
        if problem.isGoalState(front_node[0]): return path_from_root
        if front_node[0] not in closed_set:
            closed_set.append(front_node[0])
            for successor in problem.getSuccessors(front_node[0]):
                if successor[0] not in closed_set:
                    f = front_node[1] + successor[2] + heuristic(successor[0], problem)
                    fringe.push((successor[0], front_node[1] + successor[2]), f)
                    path_queue.push(path_from_root + [successor[1]], f)
        path_from_root = path_queue.pop()


class bidirectional_search:
    def __init__(self):
        self.gF = {}
        self.gB = {}
        self.hF = {}
        self.hB = {}
        self.openF = util.PriorityQueue()
        self.openB = util.PriorityQueue()
        self.closedF = {}
        self.closedB = {}

    def fF(self, node):
        return self.gF[node] + self.hF[node]

    def fB(self, node):
        return self.gB[node] + self.hB[node]

    def get_min_priorityF(self, node):
        return min(self.fF(node), 2 * self.gF[node])

    def get_max_priorityF(self, node):
        return max(self.fF(node), 2 * self.gF[node])

    def get_min_priorityB(self, node):
        return min(self.fB(node), 2 * self.gB[node])

    def get_max_priorityB(self, node):
        return max(self.fB(node), 2 * self.gB[node])

    def isGoalReached(self, posF, posB):
        if posF == posB:
            return True, "meet_in_mid"
        elif posF in self.closedB:
            return True, "inB"
        elif posB in self.closedF:  # check if the node to be expanded is already present in the ClosedF(forward) array
            return True, "inF"
        else:
            return False, ""


def MM0(problem, heuristic=nullHeuristic):
    #Since in MM0 h(n) = 0
    heuristic = nullHeuristic

    from game import Directions

    forward_node = problem.getStartState()
    backward_node = problem.goal

    bds = bidirectional_search()

    bds.gF[forward_node] = 0
    bds.gB[backward_node] = 0

    bds.hF[forward_node] = heuristic(forward_node, problem)
    bds.hB[backward_node] = heuristic(backward_node, problem)

    bds.openF.push((forward_node, [], bds.get_max_priorityF(forward_node)), bds.get_max_priorityF(forward_node))
    bds.openB.push((backward_node, [], bds.get_max_priorityB(backward_node)), bds.get_max_priorityB(backward_node))

    while not bds.openF.isEmpty() and not bds.openB.isEmpty():

        forward_node = bds.openF.pop()
        backward_node = bds.openB.pop()

        posF = forward_node[0]
        posB = backward_node[0]

        pathF = forward_node[1]
        pathB = backward_node[1]

        prminF = forward_node[2]
        prminB = backward_node[2]

        C = min(prminF, prminB)

        goal_reached, goal_condition = bds.isGoalReached(posF, posB)

        if goal_reached:
            if goal_condition == "meet_in_mid":
                return pathF + Directions.get_reverse_path(pathB)
            elif goal_condition == "inB":
                pathB = bds.closedB[posF]
                return pathF + Directions.get_reverse_path(pathB)
            elif goal_condition == "inF":
                pathF = bds.closedF[posB]
                return pathF + Directions.get_reverse_path(pathB)

        if (C == prminF):
            bds.openB.push(backward_node, prminB)
            bds.closedF[posF] = pathF
            successorsF = problem.getSuccessors(posF)
            for next_state, action, cost in successorsF:
                if bds.openF.isthere(next_state) or next_state in bds.closedF:
                    if bds.gF[next_state] <= bds.gF[posF] + cost:
                        continue
                    if bds.openF.isthere(next_state):
                        bds.openF.remove_by_value(next_state)
                    elif next_state in bds.closedF:
                        del bds.closedF[next_state]
                bds.gF[next_state] = bds.gF[posF] + cost
                bds.hF[next_state] = heuristic(next_state, problem)
                bds.openF.push((next_state, pathF + [action], bds.get_max_priorityF(next_state)),
                               bds.get_max_priorityF(next_state))

        else:
            bds.openF.push(forward_node, prminF)
            bds.closedB[posB] = pathB
            successorsB = problem.getSuccessors(posB)
            for next_state, action, cost in successorsB:
                if bds.openB.isthere(next_state) or next_state in bds.closedB:
                    if bds.gB[next_state] <= bds.gB[posB] + cost:
                        continue
                    if bds.openB.isthere(next_state):
                        bds.openB.remove_by_value(next_state)
                    elif next_state in bds.closedB:
                        del bds.closedB[next_state]
                bds.gB[next_state] = bds.gB[posB] + cost
                bds.hB[next_state] = heuristic(next_state, problem)
                bds.openB.push((next_state, pathB + [action], bds.get_max_priorityB(next_state)),
                               bds.get_max_priorityB(next_state))

    return []


def MM(problem, heuristic=nullHeuristic):
    from game import Directions

    forward_node = problem.getStartState()
    backward_node = problem.goal

    bds = bidirectional_search()

    bds.gF[forward_node] = 0
    bds.gB[backward_node] = 0

    bds.hF[forward_node] = heuristic(forward_node, problem)
    bds.hB[backward_node] = heuristic(backward_node, problem)

    bds.openF.push((forward_node, [], bds.get_max_priorityF(forward_node)), bds.get_max_priorityF(forward_node))
    bds.openB.push((backward_node, [], bds.get_max_priorityB(backward_node)), bds.get_max_priorityB(backward_node))

    while not bds.openF.isEmpty() and not bds.openB.isEmpty():

        forward_node = bds.openF.pop()
        backward_node = bds.openB.pop()

        posF = forward_node[0]
        posB = backward_node[0]

        pathF = forward_node[1]
        pathB = backward_node[1]

        prminF = forward_node[2]
        prminB = backward_node[2]

        C = min(prminF, prminB)

        goal_reached, goal_condition = bds.isGoalReached(posF, posB)

        if goal_reached:
            if goal_condition == "meet_in_mid":
                return pathF + Directions.get_reverse_path(pathB)
            elif goal_condition == "inB":
                pathB = bds.closedB[posF]
                return pathF + Directions.get_reverse_path(pathB)
            elif goal_condition == "inF":
                pathF = bds.closedF[posB]
                return pathF + Directions.get_reverse_path(pathB)

        if (C == prminF):
            bds.openB.push(backward_node, prminB)
            bds.closedF[posF] = pathF
            successorsF = problem.getSuccessors(posF)
            for next_state, action, cost in successorsF:
                if bds.openF.isthere(next_state) or next_state in bds.closedF:
                    if bds.gF[next_state] <= bds.gF[posF] + cost:
                        continue
                    if bds.openF.isthere(next_state):
                        bds.openF.remove_by_value(next_state)
                    elif next_state in bds.closedF:
                        del bds.closedF[next_state]
                bds.gF[next_state] = bds.gF[posF] + cost
                bds.hF[next_state] = heuristic(next_state, problem)
                bds.openF.push((next_state, pathF + [action], bds.get_max_priorityF(next_state)),
                               bds.get_max_priorityF(next_state))

        else:
            bds.openF.push(forward_node, prminF)
            bds.closedB[posB] = pathB
            successorsB = problem.getSuccessors(posB)
            for next_state, action, cost in successorsB:
                if bds.openB.isthere(next_state) or next_state in bds.closedB:
                    if bds.gB[next_state] <= bds.gB[posB] + cost:
                        continue
                    if bds.openB.isthere(next_state):
                        bds.openB.remove_by_value(next_state)
                    elif next_state in bds.closedB:
                        del bds.closedB[next_state]
                bds.gB[next_state] = bds.gB[posB] + cost
                bds.hB[next_state] = heuristic(next_state, problem)
                bds.openB.push((next_state, pathB + [action], bds.get_max_priorityB(next_state)),
                               bds.get_max_priorityB(next_state))

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
MM0 = MM0
MM = MM
