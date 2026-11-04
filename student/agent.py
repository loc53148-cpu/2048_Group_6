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
    # 2. Đổi tên team cho ngầu, hiển thị trên UI
    name = "Group_6" 

    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván để không vi phạm luật State"""
        clear_cache()
        try:
            init_zobrist() 
        except NameError:
            pass 

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Trả về đúng một phần tử Move có trong obs.legal_moves."""
        
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        # 3. NHIỆM VỤ CỦA TEAM LEAD: BUNG HẾT SỨC MẠNH
        if time_limit_ms > 45000.0:
            # Giai đoạn đầu: Đánh cực nhanh (15ms) để cất quỹ thời gian
            time_budget_ms = 15.0
        elif time_limit_ms > 5000.0:
            # Giai đoạn giữa (Sống còn): Bung ngân sách ra nghĩ thật sâu (400ms)
            time_budget_ms = min(400.0, time_limit_ms / 30.0)
        else:
            # Gần hết giờ: Trở về chế độ sinh tồn
            time_budget_ms = 15.0

        best_move = get_best_move_expectimax(
            board=obs.board, 
            legal_moves=obs.legal_moves, 
            time_budget_ms=time_budget_ms
        )

        return best_move
