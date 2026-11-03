"""Nhiệm vụ của Lính cát 5: Lưu trữ bộ trọng số Heuristic tốt nhất tìm được"""

# TODO: Chạy grid search và điền bộ số tốt nhất vào đây
HEURISTIC_WEIGHTS = {
    "empty": 1.0,
    "corner_penalty": 1.0,
    "monotonicity": 1.0,
    "smoothness": 1.0,
    "snake": 1.0
}