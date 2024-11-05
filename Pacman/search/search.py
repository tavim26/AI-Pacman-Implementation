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

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Returns a list of actions that reaches the goal.
    """

    # Afișează detalii inițiale despre problemă
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    from util import Stack

    # 1.init stiva start
    start_position = problem.getStartState()
    stack = Stack()
    stack.push((start_position, []))  # perechea (poz, drum pana la poz)

    # 2.set noduri nevizitate
    visited = set()

    # 3. cat timp stiva nu e goala
    while not stack.isEmpty():
        # a. extrage nod din vf stivei
        (current_position, path) = stack.pop()

        # b. daca current poz nu e dest finala, returnam drumul catre destinatie
        if problem.isGoalState(current_position):
            return path

        # c. daca current_position nu este în visited
        if current_position not in visited:
            # add current_position la visited
            visited.add(current_position)

            # d. pentru fiecare vecin al lui current_poz
            for successor, action, _ in problem.getSuccessors(current_position):

                # i. daca vecinul nu e in visited
                if successor not in visited:
                    # adauga (vecin, calea spre vecin) in stiva
                    stack.push((successor, path + [action]))


    # 4.daca stiva e goala si nu s-a gasit cale, ret null
    return []






def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the shallowest nodes in the search tree first.

    Returns a list of actions that reaches the goal.
    """

    # Afișează detalii inițiale despre problemă
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    from util import Queue

    # 1. Inițializează coada cu starea inițială
    start_position = problem.getStartState()
    queue = Queue()
    queue.push((start_position, []))  # perechea (poz, drum până la poz)

    # 2. Set de noduri vizitate
    visited = set()

    # 3. Cât timp coada nu este goală
    while not queue.isEmpty():
        # a. Extrage nodul din capul cozii
        (current_position, path) = queue.pop()

        # b. Dacă poziția curentă este destinația finală, returnează drumul către destinație
        if problem.isGoalState(current_position):
            return path

        # c. Dacă current_position nu este în visited
        if current_position not in visited:
            # Adaugă current_position la visited
            visited.add(current_position)

            # d. Pentru fiecare vecin al lui current_position
            for successor, action, _ in problem.getSuccessors(current_position):

                # i. Dacă vecinul nu este în visited
                if successor not in visited:
                    # Adaugă (vecin, calea spre vecin) în coadă
                    queue.push((successor, path + [action]))

    # 4. Dacă coada este goală și nu s-a găsit o cale, returnează null
    return []




def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()



def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0



def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()






# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
