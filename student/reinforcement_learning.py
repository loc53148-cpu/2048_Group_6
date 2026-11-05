from game2048.rules import Move
from game2048.bitboard import unpack_board, pack_board
from student.bit_rules import move_bitboard, quick_legal_moves, get_empty, spawn_tile
import numpy as np
import math
import random

class MCTSNode:
    """
    I dont know how to optimize this. Dont use it

    Parameters required tuning in this class
    - k (in self.untried_actions): Maximum no. branches of chance nodes
    - c (in __best_child): Exploration constant, used for determining a node value
    - max_depth (in simulate): Maximum no. simulation moves in rollout
    """
    __slots__ = ("board", "isBot", "parent", "children", "overall_value", "cur_score", "visits", "cause", "untried_actions")
    def __init__(self, board: int, parent: MCTSNode=None, cause: Move=None, isBot: bool=True, cur_score: int=0):
        self.board = board
        self.isBot = isBot
        self.parent = parent
        self.children = []
        self.overall_value = 0.0
        self.cur_score = cur_score
        self.visits = 0
        self.cause = cause
        self.untried_actions = None

    #Utility methods
    def __unexpandable(self):
        return self.untried_actions==[]

    def __best_child(self):
        #least_visited = min(self.children, key=lambda x: x.visits)
        #if least_visited.visits == 0:
        #    return least_visited

        def ucb1(node: MCTSNode, c: float=1.4):
            exploit = node.overall_value / node.visits
            explore = math.sqrt(math.log(self.visits)/node.visits)
            return exploit + c*explore
        return max(self.children, key=ucb1)  

    def __check_untried_actions(self):
        if self.isBot:
            available_moves = quick_legal_moves(self.board)
        else:
            empty = get_empty(self.board)
            available_moves = random.choices([[tile, number] for tile in empty for number in (1,2)], [0.9]*len(empty)+[0.1]*len(empty), k=min(6, len(empty)*2))
        random.shuffle(available_moves)
        return available_moves
    
    #Expansion
    def expand(self):
        """
        Expand from a leaf node

        Return
        ------

        Node
            The expanded child node of the leaf node
        """
        if not self.untried_actions:
            self.untried_actions = self.__check_untried_actions()
        change = self.untried_actions.pop()
        if self.isBot: 
            new_state, score = move_bitboard(self.board, change)[:2]
            child = MCTSNode(new_state, self, change, False, self.cur_score+score)
            self.children.append(child)
        else: 
            new_state = spawn_tile(self.board, change[0], change[1])
            child = MCTSNode(new_state, self, change, True, self.cur_score)
            self.children.append(child)
        return child
    
    #Simulation
    def simulate(self, max_depth=20):
        cur_state = self.board
        total_score = self.cur_score
        available_move = quick_legal_moves(self.board)
        depth = 0
        while available_move and depth<max_depth:
            move = random.choice(available_move)
            cur_state, merge_score, _changed = move_bitboard(cur_state, move)
            total_score+=merge_score
            spawn, tile = random.choice(get_empty(self.board)), (1 if random.random()<0.9 else 2)
            cur_state = spawn_tile(cur_state, spawn, tile)
            available_move = quick_legal_moves(cur_state)
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
    def search(self, iteration: int=150, evaluating: bool=True):
        for i in range(iteration):
                node = self
                while node.children and node.__unexpandable():
                    node = node.__best_child()
                if node.children!=None:
                    node = node.expand()
                score = node.simulate(25)
                node.backpropagate(score)
        try:     
            best_child = max(self.children, key= lambda x:x.visits)
        except Exception as e:
            print(e, self.children, self.untried_actions)
        return best_child.cause
