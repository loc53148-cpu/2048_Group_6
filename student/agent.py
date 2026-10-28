"""Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.

Baseline bên dưới nhìn trước đúng một nước. Nó minh họa API và cách mô phỏng
nước đi, nhưng cố ý chưa xử lý ô ngẫu nhiên và chưa phải một thuật toán mạnh.
"""

from __future__ import annotations

import numpy as np

from game2048.observation import Observation
from game2048.rules import move_board
from game2048.types import Move

class Agent:
    # Đổi tên này thành tên đội; nó sẽ xuất hiện trong UI và file kết quả.
    name = "starter-one-ply"
    tile_spawn_chance = {2: 0.9, 4: 0.1}


    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván nếu thuật toán của bạn có dùng."""

    def evaluate(self, board: np.ndarray, merge_score: int) -> float:
        """Heuristic mẫu; hãy thay bằng thiết kế của nhóm."""
        #empty_cells = int(np.count_nonzero(board == 0))
        #largest_tile = int(board.max())
        #return 250.0 * empty_cells + merge_score + 0.1 * largest_tile

        #Simple Expectimax with risk factor
        return self.maximizer(board, merge_score)

    def maximizer(self, board: np.ndarray, cum_score: int, riskiness: float =1, depth: int=0) -> float:
        if depth>=4:
            return cum_score
        max_score = float("-inf")
        for move in Move:
            projected_board, merge_score, valid = move_board(board, move)
            max_score =  max(
                            max_score, 
                            (self.expectedScore(projected_board, cum_score+merge_score, riskiness, depth+1) if valid else cum_score)
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
                    tile_score.append(self.maximizer(board, cum_score, riskiness, depth+1))
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
                        tile_score.append(self.maximizer(board, cum_score, riskiness, depth+1))
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



    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Trả về đúng một phần tử ``Move`` có trong ``obs.legal_moves``.

        Khi chấm theo tổng thời gian, ``time_limit_ms`` và
        ``obs.time_remaining_ms`` là ngân sách còn lại của cả ván — không phải
        lời hứa rằng mỗi nước được phép dùng hết số đó.
        """
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")
        best_move = obs.legal_moves[0]
        best_value = float("-inf")
        for move in obs.legal_moves:
            next_board, merge_score, _changed = move_board(obs.board, move)
            value = self.evaluate(next_board, merge_score)
            if value > best_value:
                best_value, best_move = value, move
        return best_move