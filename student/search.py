"""Nhiệm vụ của Lính cát 1: Viết thuật toán Expectimax và Chance Node Sampling"""
import numpy as np
from game2048.types import Move

def get_best_move_expectimax(board: np.ndarray, legal_moves: list[Move], time_budget_ms: float) -> Move:
    # TODO: Code cây Expectimax ở đây
    # Trả về tạm nước đi đầu tiên để không báo lỗi
    return legal_moves[0]