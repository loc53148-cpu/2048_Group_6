"""Nhiệm vụ của Lính cát 1: Viết thuật toán Expectimax và Chance Node Sampling"""
import numpy as np
from game2048.types import Move
from game2048.rules import move_board, legal_moves, is_game_over
from game2048.heuristic import monotonicity, smoothness, merge_potential, HeuristicWeights
import random
import math
class ExpectimaxExpansion:
    """Expectimax using recursive function-based"""
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
        overload_amount = 11
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
    """I dont know how to optimize this. Dont use it"""
    def __init__(self, board: np.ndarray, parent: MCTSNode=None, cause: Move=None, isBot: bool=True, cur_score: int=0):
        self.board = board
        self.isBot = isBot
        self.parent = parent
        self.children = []
        self.overall_value = 0.0
        self.cur_score = cur_score
        self.visits = 0
        self.cause = cause
        if isBot:
            available_moves = list(legal_moves(board)) 
            self.untried_actions = available_moves
            if len(available_moves)==0: 
                self.children = None
        else:
            empty = np.argwhere(board==0)
            self.untried_actions = random.choices([list(tile)+[number] for tile in empty for number in [2,4]], [0.9]*len(empty)+[0.1]*len(empty), k=min(10, len(empty)*2))

    #Utility methods
    def __unexpandable(self):
        return self.untried_actions==[]

    def __best_child(self):
        least_visited = min(self.children, key=lambda x: x.visits)
        if least_visited.visits == 0:
            return least_visited

        def ucb1(node: MCTSNode, c: float=1.4):
            exploit = node.overall_value / node.visits
            explore = math.sqrt(math.log(self.visits)/node.visits)
            return exploit + c*explore
        return max(self.children, key=ucb1)  
        
    #Expansion
    def expand(self):
        """
        Expand from a leaf node

        Return
        ------

        Node
            The expanded child node of the leaf node
        """
        change = self.untried_actions.pop(0)
        if self.isBot: 
            new_state, score = move_board(self.board, change)[:2]
            child = MCTSNode(new_state, self, change, False, self.cur_score+score)
            self.children.append(child)
        else: 
            new_state = self.board.copy()
            new_state[change[0]][change[1]] = change[2]
            child = MCTSNode(new_state, self, change, True, self.cur_score)
            self.children.append(child)
        return child
    
    #Simulation
    def simulate(self):
        cur_state = self.board.copy()
        total_score = self.cur_score
        available_move = legal_moves(cur_state)
        depth = 0
        while available_move and depth<20:
            move = random.choice(available_move)
            cur_state, merge_score, changed = move_board(cur_state, move)
            total_score+=merge_score
            if not changed:
                break
            spawn = random.choice(np.argwhere(cur_state==0))
            cur_state[spawn[0]][spawn[1]] = 2 if random.random() < 0.9 else 4
            available_move = legal_moves(cur_state)
            depth+=1
        return total_score

    #Backpropagation
    def backpropagate(self, score):
        self.visits += 1
        self.overall_value += score
        if self.parent:
            self.parent.backpropagate(score)

    #Creating tree
    #Make your own shi if you need to implement other heuristic functions, this is just base iterating method
    def search(self, iteration: int, evaluating: bool=True):
        for i in range(iteration):
            node = self
            while (node.children is not None) and node.__unexpandable():
                node = node.__best_child()
            if (node.children is not None):
                node = node.expand()
            score = node.simulate()
            node.backpropagate(score)
        best_child = max(self.children, key= lambda x:x.visits)
        return best_child.cause
#Debug template
#            try: 
#                node.<method>
#            except Exception as e:
#                print("Iteration:", i)
#                print("Metadata")
#                print(f"Children: {node.children}\nBoard: \n{node.board}\nVisits: {node.visits}\nUnexplore: {self.untried_actions}")
#                print(f"\x1b[91mFailed to <do methods>. Error msg:\x1b[0m", e)


        

    





    