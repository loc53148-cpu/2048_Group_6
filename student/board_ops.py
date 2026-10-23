"""Nhiệm vụ của Lính cát 4: Viết lại hàm move_board để chạy cực nhanh (không dùng copy)"""
import numpy as np
from game2048.types import Move

def get_empty_cells(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == 0))

# TODO: Viết hàm trượt bàn cờ siêu tốc (Fast Move Simulator)