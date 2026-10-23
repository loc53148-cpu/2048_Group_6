"""
Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.
Đây là file tổng hợp (Pipeline) do Team Lead phụ trách.
"""

from __future__ import annotations
import numpy as np

from game2048.observation import Observation
from game2048.types import Move

# ==============================================================================
# IMPORT CÁC MODULE CỦA ĐỒNG ĐỘI (Sẽ bỏ comment khi các lính cát code xong)
# ==============================================================================
# from student.search import get_best_move_expectimax
# from student.tt_cache import clear_cache
# from student.board_ops import get_empty_cells
# ==============================================================================

class Agent:
    # Tên đội sẽ xuất hiện trong UI và file kết quả benchmark
    name = "Group_6"

    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván nếu thuật toán của bạn có dùng."""
        # Gọi hàm xóa Transposition Table của Lính cát 2 ở đây để không bị đầy RAM
        # clear_cache()
        pass

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """
        Quyết định nước đi và phân bổ ngân sách 60s/ván.
        """
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        # ----------------------------------------------------------------------
        # BƯỚC 1: TÍNH TOÁN NGÂN SÁCH THỜI GIAN CHO NƯỚC ĐI NÀY
        # ----------------------------------------------------------------------
        # Đếm số ô trống (Tạm thời dùng numpy, sau này dùng hàm siêu tốc của Lính cát 4)
        empty_cells = int(np.count_nonzero(obs.board == 0))
        
        # Chiến thuật chia thời gian cực kỳ quan trọng:
        if empty_cells >= 10:
            # Đầu game, bàn rất thoáng: Đi thật nhanh (cho phép 20ms)
            time_for_this_move = 20.0
        elif empty_cells >= 5:
            # Giữa game: Trích một phần nhỏ quỹ thời gian còn lại
            time_for_this_move = time_limit_ms / 250.0
        elif empty_cells >= 3:
            # Bàn khá chật: Bắt đầu nghĩ sâu hơn
            time_for_this_move = time_limit_ms / 50.0
        else:
            # Ngàn cân treo sợi tóc (<3 ô trống): Dồn nhiều thời gian để gỡ bí
            time_for_this_move = time_limit_ms / 15.0
            
        # Đặt ngưỡng an toàn để không bao giờ xài lố quá 1 giây cho 1 nước
        time_for_this_move = min(time_for_this_move, 1000.0)
        
        # ----------------------------------------------------------------------
        # BƯỚC 2: GỌI THUẬT TOÁN TÌM KIẾM CỦA TEAM (EXPECTIMAX + TT CACHE)
        # ----------------------------------------------------------------------
        # Khi Lính cát 1 & 2 code xong, bạn bỏ comment đoạn dưới đây để chạy thật:
        
        """
        best_move = get_best_move_expectimax(
            board=obs.board,
            legal_moves=obs.legal_moves,
            time_budget_ms=time_for_this_move
        )
        return best_move
        """
        
        # [TẠM THỜI]: Trả về nước đi đầu tiên hợp lệ để file không bị báo lỗi 
        # trong lúc chờ đồng đội code xong các file khác.
        return obs.legal_moves[0]
