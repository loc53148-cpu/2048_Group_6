"""Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.

Baseline bên dưới nhìn trước đúng một nước. Nó minh họa API và cách mô phỏng
nước đi, nhưng cố ý chưa xử lý ô ngẫu nhiên và chưa phải một thuật toán mạnh.
"""
from __future__ import annotations
import numpy as np
import random
import math

from game2048.observation import Observation
from game2048.rules import move_board
from game2048.types import Move

# 1. Kéo vũ khí hạng nặng của team vào đây
from student.search import get_best_move_expectimax
from student.tt_cache import clear_cache

class Agent:
    # 2. Đổi tên team , hiển thị trên UI
    name = "Group_6" 

    def __init__(self):
        # Các biến trạng thái được benchmark.py và kehoach.md đọc để báo cáo số liệu thực tế
        self.last_completed_depth: int = 0  # Benchmark đọc trường này
        self.search_depth: int = 0          # Kế hoạch kehoach.md yêu cầu trường này
        self.last_nodes: int = 0            # Số node đã duyệt (tùy chọn)

    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván để không vi phạm luật State"""
        clear_cache()
        self.last_completed_depth = 0
        self.search_depth = 0
        self.last_nodes = 0
        # Gọi hàm tạo lại ma trận băm ngẫu nhiên của Phương (nếu Phương đã sửa theo yêu cầu)
        try:
            init_zobrist() 
        except NameError:
            pass # Phòng hờ Phương chưa kịp viết hàm này thì game vẫn chạy được

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Trả về đúng một phần tử Move có trong obs.legal_moves."""
        
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        # 3. QUẢN LÝ NGÂN SÁCH THỜI GIAN THÔNG MINH:
        is_per_game_budget = (obs.total_time_limit_ms is not None) or (time_limit_ms > 500)

        if not is_per_game_budget:
            # CHẾ ĐỘ A: Giới hạn theo từng nước (Per-move cap, VD: 50ms)
            # Chừa đệm 4ms an toàn tuyệt đối để đảm bảo timeouts: 0
            time_budget_ms = max(5.0, time_limit_ms - 4.0)
        else:
            # CHẾ ĐỘ B: Tổng thời gian cả ván (Total-budget, VD: 60.000ms đếm lùi)
            time_budget_ms = min(80.0, time_limit_ms / 30.0)
            if time_limit_ms < 1500:
                time_budget_ms = 8.0
            elif time_limit_ms < 4000:
                time_budget_ms = 18.0

        # Đảm bảo luôn có tối thiểu 2.5ms
        time_budget_ms = max(2.5, time_budget_ms)

        # 4. Truyền bàn cờ, các nước đi hợp lệ, ngân sách thời gian và agent_instance xuống
        best_move = get_best_move_expectimax(
            board=obs.board, 
            legal_moves=obs.legal_moves, 
            time_budget_ms=time_budget_ms,
            agent_instance=self
        )

        return best_move
