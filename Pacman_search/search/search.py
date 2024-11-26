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

from django.contrib.admin import action

import util
from game import Directions
from typing import List

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




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]




# things to be implemented below



# DFS
def depthFirstSearch(problem: SearchProblem) -> List[Directions]:

    stack = [(problem.getStartState(), [])]

    visited = set()

    while stack:
        current_position, path = stack.pop()

        if problem.isGoalState(current_position):
            return path

        if current_position not in visited:
            visited.add(current_position)

            for successor, action, cost in problem.getSuccessors(current_position):
                if successor not in visited:
                    stack.append((successor, path + [action]))

    return []




#BFS
def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:

    from util import Queue

    start_position = problem.getStartState()
    queue = Queue()
    queue.push((start_position, []))

    visited = set()

    while not queue.isEmpty():
        current_position, path = queue.pop()

        if problem.isGoalState(current_position):
            return path

        if current_position not in visited:
            visited.add(current_position)

            for successor, action, _ in problem.getSuccessors(current_position):
                if successor not in visited:
                    queue.push((successor, path + [action]))

    return []



#uniform cost search
def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the node of least total cost first.

    Returns a list of actions that reaches the goal.
    """

    from util import PriorityQueue

    start_position = problem.getStartState()
    priority_queue = PriorityQueue()
    priority_queue.push((start_position, []), 0)

    cost_map = {start_position: 0}

    visited = set()

    while not priority_queue.isEmpty():
        current_position, path = priority_queue.pop()

        if problem.isGoalState(current_position):
            return path

        if current_position not in visited:
            visited.add(current_position)

            for successor, action, step_cost in problem.getSuccessors(current_position):
                new_cost = cost_map[current_position] + step_cost

                if successor not in visited or new_cost < cost_map.get(successor, float('inf')):
                    cost_map[successor] = new_cost
                    priority_queue.push((successor, path + [action]), new_cost)

    return []




def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


# A Star search
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:


    from util import PriorityQueue

    start_state = problem.getStartState()
    priority_queue = PriorityQueue()
    priority_queue.push((start_state, []), heuristic(start_state, problem))

    cost_map = {start_state: 0}

    while not priority_queue.isEmpty():
        current_state, path = priority_queue.pop()

        if problem.isGoalState(current_state):
            return path

        for successor, action, step_cost in problem.getSuccessors(current_state):
            new_cost = cost_map[current_state] + step_cost

            if successor not in cost_map or new_cost < cost_map[successor]:
                cost_map[successor] = new_cost
                priority = new_cost + heuristic(successor, problem)
                priority_queue.push((successor, path + [action]), priority)

    return []




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
