# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """




    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]




    def evaluationFunction(self, currentGameState: GameState, action):
        """
        A better evaluation function for ReflexAgent.

        Takes into account food, ghost positions, and Pacman's position.
        Rewards Pacman for getting closer to food, and penalizes for being near ghosts.
        """
        # Get the successor game state after Pacman takes the action
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Pacman's new position after taking the action
        newPos = successorGameState.getPacmanPosition()

        # List of all the food locations
        newFood = successorGameState.getFood()
        foodList = newFood.asList()

        # List of all the ghost states
        newGhostStates = successorGameState.getGhostStates()

        # Times each ghost is scared (due to power pellets)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Calculate the score based on the successor game state
        score = successorGameState.getScore()

        # Reward for getting closer to food (using the minimum distance to any food)
        foodDistance = float('inf')
        if len(foodList) > 0:
            foodDistance = min([manhattanDistance(newPos, food) for food in foodList])

        # Penalize for being closer to ghosts that are not scared
        ghostDistance = float('inf')
        ghostPenalty = 0
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0:  # Ghosts that are not scared
                if distanceToGhost <= 1:
                    ghostPenalty = 100  # Large penalty if Pacman is too close to a non-scared ghost
                else:
                    ghostDistance = min(ghostDistance, distanceToGhost)

        # Reward for ghosts being scared (avoid getting too close)
        scaredReward = 0
        for scaredTime in newScaredTimes:
            if scaredTime > 0:  # Ghost is scared
                scaredReward += 5  # Slight reward for being close to scared ghosts (for capturing them)

        # Combine all the factors:
        # - Positive reward for getting closer to food (inverse of food distance)
        # - Negative reward for being close to non-scared ghosts (penalty based on ghost distance)
        # - Positive reward for being near scared ghosts (if any)
        foodScore = -foodDistance if foodDistance < float('inf') else 0
        ghostScore = -ghostPenalty if ghostPenalty > 0 else (-2 * ghostDistance if ghostDistance < float('inf') else 0)

        # Add the final score: base score, food score, ghost penalty, and scared ghost reward
        finalScore = score + foodScore + ghostScore + scaredReward

        return finalScore




def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()




class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)



class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """



    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()




def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()




# Abbreviation
better = betterEvaluationFunction
