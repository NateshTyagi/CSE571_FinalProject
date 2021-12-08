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


def MM0(problem):
    def PathReverse(p):
        """
        Given a action list, return the reversed version of it for the nodes expanding in the backward direction.
        """
        path = []
        for x in p:
            if x == 'North':
                z = 'South'
                path.append(z)
            if x == 'South':
                z = 'North'
                path.append(z)
            if x == 'West':
                z = 'East'
                path.append(z)
            if x == 'East':
                z = 'West'
                path.append(z)
        return path[::-1]

    gF = 0  # initialize gF value to be 0 as the cost of returning to the start state from the start node is zero

    gB = 0
    OpenF = util.PriorityQueue()  # create a Priority Queue to store all the nodes expanded in the forward direction
    OpenB = util.PriorityQueue()  # craete a Priority Queue to store all the nodes expanded in the backward direction
    OpenF.push((problem.getStartState(), [], 0),
               2 * gF)  # push the start state to the Queue.Since the heuristic value is zero we take into account the the value of max(g,2*g) = 2*g
    OpenB.push((problem.goal, [], 0), 2 * gB)  # push the goal state to the Queue

    ClosedF = {}  # dictionary to store the path to reach the node from start node, with the key being its location
    ClosedB = {}  # dictionary to store the path to reach the node from goal node, with the key being its location
    gF_dic = {}  # dictionary to store the cost to reach the node from start node, with the key being its location
    gB_dic = {}  # dictionary to store the cost to reach the node from goal node, with the key being its location
    gF_dic[problem.getStartState()] = gF
    gB_dic[problem.goal] = gB
    U = float('inf')

    while (not OpenF.isEmpty()) and (not OpenB.isEmpty()):

        CurrentPopF = OpenF.pop()
        CurrentPopB = OpenB.pop()
        StateF = CurrentPopF[0]
        StateB = CurrentPopB[0]
        gF = CurrentPopF[2]
        gB = CurrentPopB[2]
        pathF = CurrentPopF[1]
        pathB = CurrentPopB[1]

        C = min(gF, gB)  # find the minimum cost value (i.e from the forward node and backward node)

        if StateF == StateB:
            print('reached goal1')
            return pathF + PathReverse(pathB)
        if StateF in ClosedB:
            pathB = ClosedB[StateF]
            print('reached goal2')
            return pathF + PathReverse(pathB)
        if StateB in ClosedF:
            pathF = ClosedF[StateB]
            print('reached goal3')
            return pathF + PathReverse(pathB)

        if (
                C == gF):  # If the cost of expanding a node in the forward iteration is lesser, then expand node in forward direction
            OpenB.push(CurrentPopB, 2 * gB)  # Push back the popped node of backward iteration in the queue
            ClosedF[CurrentPopF[
                0]] = pathF  # store the popped node's path in dictionary closedF with key as the node's location
            SuccessorsF = problem.getSuccessors(StateF)
            for i in SuccessorsF:
                if OpenF.isthere(i[0]) or i[
                    0] in ClosedF:  # check if successor is already present in OpenF or in ClosedF(i.e. already visited nodes)
                    if gF_dic[i[0]] < gF + i[
                        2]:  # If yes, check if this node's stored cost is less than sum of cost to current node + cost of edge to the successor node
                        continue
                    if OpenF.isthere(i[0]):
                        OpenF.remove_by_value(i[
                                                  0])  # Remove node from OpenF queue if the successor node is present there. Check function in util.py.
                    elif i[0] in ClosedF:
                        del ClosedF[i[0]]  # remove node from ClosedF if the successor node is present there
                gF_dic[i[0]] = gF + i[2]  # update the cost to reach the succesor node and then push it to the queue
                OpenF.push((i[0], pathF + [i[1]], gF + i[2]), 2 * (gF + i[2]))

        else:
            OpenF.push(CurrentPopF, 2 * gF)
            ClosedB[CurrentPopB[0]] = pathB
            SuccessorsB = problem.getSuccessors(StateB)
            for i in SuccessorsB:
                if OpenB.isthere(i[0]) or i[0] in ClosedB:
                    if gB_dic[i[0]] < gB + i[2]:
                        continue
                    if OpenB.isthere(i[0]):
                        OpenB.remove_by_value(i[0])
                    elif i[0] in ClosedB:
                        del ClosedB[i[0]]
                gB_dic[i[0]] = gB + i[2]
                OpenB.push((i[0], pathB + [i[1]], gB + i[2]), 2 * (gB + i[2]))

    return []


def MM(problem, heuristic=nullHeuristic):
    def PathReverse(p):
        """
        Given a action list, return the reversed version of it for the nodes expanded in the backward direction.
        """
        path = []
        for x in p:
            if x == 'North':
                z = 'South'
                path.append(z)
            if x == 'South':
                z = 'North'
                path.append(z)
            if x == 'West':
                z = 'East'
                path.append(z)
            if x == 'East':
                z = 'West'
                path.append(z)
        return path[::-1]

    gF = 0
    epsilon = 1
    gB = 0
    OpenF = util.PriorityQueue()
    OpenB = util.PriorityQueue()
    hf = heuristic(problem.getStartState(), problem)
    hb = heuristic(problem.goal, problem)
    OpenF.push((problem.getStartState(), [], 0), max(hf, 2 * gF))
    OpenB.push((problem.goal, [], 0), max(hb, 2 * gB))

    ClosedF = {}  # dictionary to store the path to reach the node from start node, with the key being its location
    ClosedB = {}  # dictionary to store the cost to reach the node from goal node, with the key being its location
    gF_dic = {}  # dictionary to store the cost to reach the node from start node, with the key being its location
    gB_dic = {}  # dictionary to store the cost to reach the node from goal node, with the key being its location
    gF_dic[problem.getStartState()] = gF
    gB_dic[problem.goal] = gB

    while (not OpenF.isEmpty()) and (not OpenB.isEmpty()):

        CurrentPopF = OpenF.pop()
        CurrentPopB = OpenB.pop()
        StateF = CurrentPopF[0]
        StateB = CurrentPopB[0]
        gF = CurrentPopF[2]
        gB = CurrentPopB[2]
        pathF = CurrentPopF[1]
        pathB = CurrentPopB[1]

        C = min(gF,
                gB)  # check expnding which node is less costlier, i.e. node in the forward direction or the node in backward direction.

        if StateF == StateB:  # check if the current nodes of forward and backward are meeting
            print('reached goal1')
            return pathF + PathReverse(pathB)
        if StateF in ClosedB:  # check if the node to be expanded is alredy present in the array which stores the node expanded by the backward search
            pathB = ClosedB[StateF]
            print('reached goal2')
            return pathF + PathReverse(pathB)
        if StateB in ClosedF:  # check if the node to be expanded is already present in the ClosedF(forward) array
            pathF = ClosedF[StateB]
            print('reached goal3')
            return pathF + PathReverse(pathB)

        if (
                C == gF):  # If the cost of expanding a node in the forward iteration is lesser, then expand node in forward direction
            OpenB.push(CurrentPopB, gB)  # Push back the popped node of backward iteration in the queue
            ClosedF[CurrentPopF[
                0]] = pathF  # store the popped node's path in dictionary closedF with key as the node's location
            SuccessorsF = problem.getSuccessors(StateF)
            for i in SuccessorsF:
                h_f = heuristic(i[0], problem)
                if OpenF.isthere(i[0]) or i[
                    0] in ClosedF:  # check if successor is already present in OpenF or in ClosedF(i.e. already visited nodes)
                    if gF_dic[i[0]] < gF + i[
                        2]:  # If yes, check if this node's stored cost is less than sum of cost to current node + cost of edge to the successor node
                        continue
                    if OpenF.isthere(i[0]):
                        OpenF.remove_by_value(i[
                                                  0])  # Remove node from OpenF queue if the successor node is present there. Check function in util.py.
                    elif i[0] in ClosedF:
                        del ClosedF[i[0]]  # remove node from ClosedF if the successor node is present there
                gF_dic[i[0]] = gF + i[2]  # update the cost to reach the succesor node and then push it to the queue
                ff = h_f + gF + i[2]  # f(x) = g(x) + h(x)
                OpenF.push((i[0], pathF + [i[1]], max(ff, 2 * (gF + i[2]))), max(ff, 2 * (
                        gF + i[2])))  # choose the cost value which satisfies max(f(x),2*g(x)) as the priority value

                # if OpenB.isthere(i[0]):
                #     U = min(U, gF_dic[i[0]] + gB_dic[1[0]])
        else:
            OpenF.push(CurrentPopF, gF)
            ClosedB[CurrentPopB[0]] = pathB
            SuccessorsB = problem.getSuccessors(StateB)
            for i in SuccessorsB:
                h_b = heuristic(i[0], problem)
                if OpenB.isthere(i[0]) or i[0] in ClosedB:
                    if gB_dic[i[0]] < gB + i[2]:
                        continue
                    if OpenB.isthere(i[0]):
                        OpenB.remove_by_value(i[0])
                    elif i[0] in ClosedB:
                        del ClosedB[i[0]]
                gB_dic[i[0]] = gB + i[2]
                fb = h_b + gB + i[2]
                OpenB.push((i[0], pathB + [i[1]], max(fb, 2 * (gB + i[2]))), max(fb, 2 * (gB + i[2])))

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bd0 = MM0
bd = MM
