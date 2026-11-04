"""Nhiệm vụ của Lính cát 3: Viết hàm đánh giá bàn cờ (Snake, Monotonicity, Smoothness)"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    """Tách riêng các hệ số để dễ dàng chạy script tinh chỉnh (QA)."""
    empty: float = 270.0
    monotonicity: float = 47.0
    smoothness: float = 15.0
    snake: float = 1.0
    corner_penalty: float = 10000.0

_DEFAULT_SNAKE_MATRIX = np.array([
    [32768, 16384, 8192, 4096],
    [256,   512,   1024, 2048],
    [128,   64,    32,   16],
    [1,     2,     4,    8]
], dtype=np.float64)

def get_snake_matrix() -> np.ndarray:
    return _DEFAULT_SNAKE_MATRIX

def count_empty_cells(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == 0))

def compute_monotonicity(board: np.ndarray) -> float:
    exponents = np.zeros(board.shape, dtype=np.float64)
    occupied = board > 0
    exponents[occupied] = np.log2(board[occupied])
    total = 0.0
    for lines in (exponents, exponents.T):
        for line in lines:
            increasing = float(np.maximum(line[1:] - line[:-1], 0).sum())
            decreasing = float(np.maximum(line[:-1] - line[1:], 0).sum())
            total += max(increasing, decreasing)
    return total

def compute_smoothness(board: np.ndarray) -> float:
    exponents = np.zeros(board.shape, dtype=np.float64)
    occupied = board > 0
    exponents[occupied] = np.log2(board[occupied])

    horizontal = (exponents[:, :-1] > 0) & (exponents[:, 1:] > 0)
    vertical = (exponents[:-1, :] > 0) & (exponents[1:, :] > 0)

    penalty = 0.0
    penalty += float(np.abs(exponents[:, :-1] - exponents[:, 1:])[horizontal].sum())
    penalty += float(np.abs(exponents[:-1, :] - exponents[1:, :])[vertical].sum())
    return penalty

def evaluate_board(board: np.ndarray, weights: HeuristicWeights = HeuristicWeights()) -> float:
    empty_score = float(count_empty_cells(board))
    snake_matrix = get_snake_matrix()
    snake_score = float(np.sum(board * snake_matrix))

    max_val = board.max()
    corner_penalty = 0.0
    if max_val > 0:
        corners = [board[0, 0], board[0, -1], board[-1, 0], board[-1, -1]]
        if max_val not in corners or board[0, 0] != max_val:
            corner_penalty = 1.0

    monotonicity_score = compute_monotonicity(board)
    smoothness_score = compute_smoothness(board)

    total_score = (
        weights.empty * empty_score
        + weights.snake * snake_score
        + weights.monotonicity * monotonicity_score
        - weights.smoothness * smoothness_score
        - weights.corner_penalty * corner_penalty
    )

    return float(total_score)
