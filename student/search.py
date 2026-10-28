"""Nhiệm vụ của Lính cát 1: Viết thuật toán Expectimax và Chance Node Sampling"""
import numpy as np
from game2048.types import Move
from game2048.rules import move_board

class ExpectimaxExpansion:
    def __init__(self, tile_spawn_chance: dict[int, float]={2: 0.9, 4: 0.1}, riskiness: float=1):
        self.tile_spawn_chance = tile_spawn_chance
        self.riskiness = riskiness


    def maximizer(self, board: np.ndarray, cum_score: int=0, depth: int=0) -> float:
        if depth>=4:
            return cum_score
        max_score = float("-inf")
        for move in Move:
            projected_board, merge_score, valid = move_board(board, move)
            max_score =  max(
                            max_score, 
                            (self.expectedScore(projected_board, cum_score+merge_score, self.riskiness, depth+1) if valid else cum_score)
                            )
        return max_score
        

    def expectedScore(self, board: np.ndarray, cum_score: int, riskiness: float, depth: int) -> float:
        """
        Calculated estimated value of current board, over all possible variants of current board *(Possible states after engine randomly spawns new tile)*

        Parameters
        ----------
        board: numpy array
            Current state of the game
        cum_score: int
            The simulated score of the current state of the game (Score from merging)
        riskiness: float
            Risky factor to reduce the weigh(spawn probability) used in calculating mean score of states with small scores.
            Small scores are scores that have smaller than the initially calculated mean score before add in risk factor.
        depth: int
            Current depth of the recursion 

        Return
        ------
        float
            The final calculated mean score
        """
        tile_score = []
        tile_chance = []
        row, col = board.shape
        overload_amount = 9
        #Loop through all empty cells and calculate the value of each possible spawn
        #Consider only states with 2 as the possible spawn tile if there is more than 9 empty square to cut down computation cost
        if board[board==0].sum()>overload_amount:
            for i in range(row):
                for j in range(col):
                    value = board[i][j]
                    if value>0:
                        continue
                    board[i][j] = 2
                    tile_score.append(self.maximizer(board, cum_score, depth+1))
                    tile_chance.append(1)
                    board[i][j] = 0
        else: 
            for i in range(row):
                for j in range(col):
                    value = board[i][j]
                    if value>0:
                        continue
                    for new_tile in self.tile_spawn_chance:
                        board[i][j] = new_tile
                        tile_score.append(self.maximizer(board, cum_score, depth+1))
                        tile_chance.append(self.tile_spawn_chance[new_tile])
                    board[i][j] = 0

        #Changing to numpy array for faster calculation
        #Divide all values in title_chance by half the number of empty cells so that total probability add up to 1
        tile_score = np.array(tile_score)
        tile_chance = np.array(tile_chance)/(len(tile_chance)/2) 

        #Divide all scores less than pre-adjusted mean score by riskiness parameter
        #Scale other scores up individually equal so that total probability add up to 1, 
        #and the relative size between them doesnt change
        initial_mean_score = (tile_score*tile_chance).sum()

        small_score = tile_score <= initial_mean_score

        #Change score according to riskiness parameter only if there are branches with score bigger than mean score
        if sum(small_score)==len(tile_score):
            return initial_mean_score
        tile_chance[small_score] /= riskiness
        total_small_score_chance = tile_chance[small_score].sum()
        tile_chance[~small_score] *= (1-total_small_score_chance/riskiness)/(1-total_small_score_chance)

        return (tile_score*tile_chance).sum()

class MCTSNode:
    def __init__(self, board: np.ndarray, parent=None, cause: Move=None, isBot: bool=True):
        self.board = board
        self.parent = parent
        self.children = []
        self.simulated_value = 0.0
        self.visits = 0
        self.cause = cause
        self.untried_actions = [move for move in Move if move_board(board, move)[2]] if isBot else np.argwhere(board==0)

    #Utility methods
    def __is_terminal(self):
        pass

    