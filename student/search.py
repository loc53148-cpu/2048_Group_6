"""Nhiệm vụ phối hợp của Lính cát 1 & 2: Viết thuật toán Expectimax, Quản lý thời gian và Cache"""
import time 
import numpy as np
import random
import math
from game2048.types import Move
from game2048.rules import move_board, legal_moves, is_game_over
from game2048.heuristic import monotonicity, smoothness, merge_potential, HeuristicWeights

# Import các hàm quản lý bộ nhớ đệm do Phương viết từ file tt_cache.py
from student.tt_cache import lookup_cache, store_cache

class ExpectimaxExpansion:
    """Expectimax sử dụng giải thuật đệ quy tích hợp Quản lý thời gian & Cache"""
    def __init__(self, tile_spawn_chance: dict[int, float]={2: 0.9, 4: 0.1}, riskiness: float=1):
        self.tile_spawn_chance = tile_spawn_chance
        self.riskiness = riskiness

    def maximizer(self, board: np.ndarray, cum_score: int=0, depth: int=0, start_time: float=0.0, time_limit_ms: float=60000.0, max_depth: int=4) -> float:
        # BẢO VỆ KHẨN CẤP (Nhiệm vụ của Phương): Ngắt đệ quy ngay nếu sắp lố thời gian nước đi (chừa hao 1.5ms)
        if (time.time() - start_time) * 1000 > (time_limit_ms - 1.5):
            return float(cum_score)

        # Điều kiện dừng thay đổi động theo độ sâu tối đa của vòng lặp Iterative Deepening
        if depth >= max_depth:
            return float(cum_score)
            
        max_score = float("-inf")
        for move in Move:
            projected_board, merge_score, valid = move_board(board, move)
            
            # Truyền thêm các tham số start_time, time_limit_ms, max_depth vào các tầng đệ quy kế tiếp
            score = self.expectedScore(projected_board, cum_score+merge_score, self.riskiness, depth+1, start_time, time_limit_ms, max_depth) if valid else float(cum_score)
            if score > max_score:
                max_score = score
                
        return max_score

    def expectedScore(self, board: np.ndarray, cum_score: int, riskiness: float, depth: int, start_time: float=0.0, time_limit_ms: float=60000.0, max_depth: int=4) -> float:
        # BẢO VỆ KHẨN CẤP (Nhiệm vụ của Phương): Kiểm tra thời gian trong tầng Chance Node
        if (time.time() - start_time) * 1000 > (time_limit_ms - 1.5):
            return float(cum_score)

        tile_score = []
        tile_chance = []
        row, col = board.shape
        overload_amount = 11
        
        if board[board==0].sum() > overload_amount:
            for i in range(row):
                for j in range(col):
                    value = board[i][j]
                    if value > 0:
                        continue
                    board[i][j] = 2
                    tile_score.append(self.maximizer(board, cum_score, depth+1, start_time, time_limit_ms, max_depth))
                    tile_chance.append(1)
                    board[i][j] = 0
        else:
            for i in range(row):
                for j in range(col):
                    value = board[i][j]
                    if value > 0:
                        continue
                    for new_tile in self.tile_spawn_chance:
                        board[i][j] = new_tile
                        tile_score.append(self.maximizer(board, cum_score, depth+1, start_time, time_limit_ms, max_depth))
                        tile_chance.append(self.tile_spawn_chance[new_tile])
                        board[i][j] = 0
                        
        if not tile_score:
            return float(cum_score)

        tile_score = np.array(tile_score)
        tile_chance = np.array(tile_chance)/(len(tile_chance)/2)
        
        initial_mean_score = (tile_score * tile_chance).sum()
        small_score = tile_score <= initial_mean_score
        
        if sum(small_score) == len(tile_score):
            return initial_mean_score
            
        tile_chance[small_score] /= riskiness
        total_small_score_chance = tile_chance[small_score].sum()
        tile_chance[~small_score] *= (1 - total_small_score_chance / riskiness) / (1 - total_small_score_chance)
        
        return float((tile_score * tile_chance).sum())


# =====================================================================
# 🚀 PHẦN CODE CHÍNH CỦA PHƯƠNG: ĐIỀU PHỐI THỜI GIAN, SÂU DẦN & CACHE
# =====================================================================
def get_best_move_expectimax(board: np.ndarray, legal_moves: list[Move], time_budget_ms: float) -> Move:
    """
    Hàm giao tiếp chính nhận dữ liệu từ agent.py để tìm nước đi tối ưu nhất.
    Tích hợp chiến lược Phân bổ thời gian (Ô màu vàng), Iterative Deepening và Cache.
    """
    if not legal_moves:
        return Move.UP
        
    start_time = time.time()
    best_move = legal_moves[0]  # Nước đi dự phòng khẩn cấp nếu hết giờ ngay lập tức
    
    # 1. CHIẾN LƯỢC PHÂN BỔ QUỸ THỜI GIAN (Chiến lược "Ô màu vàng")
    empty_count = np.count_nonzero(board == 0)
    if empty_count > 10:
        # Bàn cờ thoáng (>10 ô trống): Đi cực nhanh trong tối đa 15ms, giới hạn độ sâu nông
        effective_time_limit_ms = min(15.0, time_budget_ms)
        max_depth_limit = 3
    else:
        # Bàn cờ bí: Bung toàn bộ thời gian hệ thống cho phép để tính toán rất sâu
        effective_time_limit_ms = time_budget_ms
        max_depth_limit = 12

    # Khởi tạo đối tượng cây của Minh để bắt đầu tìm kiếm
    ai_engine = ExpectimaxExpansion()

    # 2. VÒNG LẶP SÂU DẦN (Iterative Deepening từ depth = 1 -> max_depth_limit)
    for current_max_depth in range(1, max_depth_limit + 1):
        # Kiểm tra thời gian trước khi tăng thêm một tầng độ sâu mới
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > (effective_time_limit_ms - 2.0):  # Chừa hao 2ms an toàn để kịp quay đầu
            break

        # 3. TỐI ƯU SẮP XẾP NƯỚC ĐI BẰNG CACHE (Move Ordering)
        # Tra cứu xem thế cờ này từng tính chưa để ưu tiên duyệt hướng đi tốt nhất cũ lên đầu
        ordered_moves = legal_moves.copy()
        cached_res = lookup_cache(board)
        if cached_res:
            cached_depth, cached_score, cached_move = cached_res
            if cached_move in ordered_moves:
                ordered_moves.remove(cached_move)
                ordered_moves.insert(0, cached_move)

        current_depth_best_move = None
        max_score = -float('inf')

        # 4. DUYỆT TÌM NƯỚC ĐI TỐT NHẤT TRONG TẦNG ĐỘ SÂU HIỆN TẠI
        for move in ordered_moves:
            projected_board, merge_score, valid = move_board(board, move)
            if not valid:
                continue
                
            # Gọi tầng đệ quy tính toán của Minh, truyền kèm các tham số đo thời gian thực tế
            score = ai_engine.expectedScore(
                projected_board, 
                cum_score=merge_score, 
                riskiness=ai_engine.riskiness, 
                depth=1, 
                start_time=start_time, 
                time_limit_ms=effective_time_limit_ms,
                max_depth=current_max_depth
            )
            
            if score > max_score:
                max_score = score
                current_depth_best_move = move

        # 5. CHỐT KẾT QUẢ TẦNG: Chỉ lưu vào cache và ghi nhận nước đi nếu tính trọn vẹn tầng và kịp giờ
        current_elapsed_ms = (time.time() - start_time) * 1000
        if current_depth_best_move is not None and current_elapsed_ms < effective_time_limit_ms:
            best_move = current_depth_best_move
            store_cache(board, current_max_depth, max_score, best_move)

    return best_move
