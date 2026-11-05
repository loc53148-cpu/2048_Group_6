"""
Nhiệm vụ phối hợp của Linh cát 1 & 2: Viết thuật toán Expectimax, Quản lý thời gian và Cache
Phiên bản tối ưu hóa cấu trúc thế trận và tốc độ thực thi cho Group_6
"""
import time
import random
import numpy as np
import math
from game2048.types import Move
from game2048.heuristic import monotonicity, smoothness, merge_potential, HeuristicWeights
from game2048.bitboard import pack_board, unpack_board
from student.bit_rules import move_bitboard

# ĐỒNG BỘ LINH CÁT 2: Quản lý bộ nhớ đệm do Phương viết từ file tt_cache.py
from student.tt_cache import lookup_cache, store_cache
# ĐỒNG BỘ LINH CÁT 3: Hàm đánh giá thế cờ dồn góc từ file heuristic.py
from student.heuristic import evaluate_board

class ExpectimaxExpansion:
    """Expectimax sử dụng giải thuật đệ quy tích hợp Quản lý thời gian & Cache"""
    def __init__(self, tile_spawn_chance: dict = {2: 0.9, 4: 0.1}, riskiness: float = 1.0):
        self.tile_spawn_chance = tile_spawn_chance
        self.riskiness = riskiness

    def maximizer(self, board: int, cum_score: float = 0.0, depth: int = 0, start_time: float = 0.0, time_limit_ms: float = 60000.0, max_depth: int = 4) -> float:
        # BẢO VỆ KHẨN CẤP: Ngắt đệ quy ngay nếu sắp lố thời gian nước đi (chừa hao 1.5ms)
        if (time.time() - start_time) * 1000 > (time_limit_ms - 1.5):
            return cum_score + evaluate_board(unpack_board(board))

        # ĐIỀU KIỆN DỪNG: Đạt độ sâu tối đa -> Trả về điểm gộp thực tế cộng điểm đánh giá Heuristic thế trận
        if depth >= max_depth:
            return cum_score + evaluate_board(unpack_board(board))

        max_score = float("-inf")
        # Duyệt qua các hướng đi hợp lệ
        for move in Move:
            projected_board, merge_score, valid = move_bitboard(board, move)
            if not valid:
                continue
            
            # CÂN BẰNG CHIẾN THUẬT: Giảm trọng số điểm ăn thô (merge_score * 0.05) để làm tiêu chí phụ phá vỡ thế cân bằng.
            # Ép Bot bắt buộc phải tập trung tối đa giữ vững cấu trúc dồn góc mượt mà từ hàm evaluate_board.
            score = self.expectedScore(projected_board, cum_score + (merge_score * 0.05), self.riskiness, depth + 1, start_time, time_limit_ms, max_depth)
            if score > max_score:
                max_score = score
                
        if max_score == float("-inf"):
            return cum_score + evaluate_board(unpack_board(board))
            
        return max_score

    def expectedScore(self, board: np.ndarray, cum_score: float, riskiness: float, depth: int, start_time: float, time_limit_ms: float = 60000.0, max_depth: int = 4) -> float:
        # BẢO VỆ KHẨN CẤP: Kiểm tra thời gian trong tầng Chance Node
        if (time.time() - start_time) * 1000 > (time_limit_ms - 1.5):
            return cum_score + evaluate_board(board)
            
        if depth >= max_depth:
            return cum_score + evaluate_board(board)

        tile_score = []
        tile_chance = []
        row, col = board.shape
        overload_amount = 11

        # SỬA LỖI ĐẾM Ô TRỐNG CHÍ MẠNG TRÊN LỚP MẢNG BOOLEAN GỐC:
        # Thay thế mảng boolean `.sum()` chạy sai bằng hàm đếm phần tử numpy chuẩn xác
        empty_tiles_count = np.count_nonzero(board == 0)

        if empty_tiles_count > overload_amount:
            for i in range(row):
                for j in range(col):
                    if board[i, j] > 0:
                        continue
                    board[i, j] = 2
                    tile_score.append(self.maximizer(board, cum_score, depth + 1, start_time, time_limit_ms, max_depth))
                    tile_chance.append(1.0)
                    board[i, j] = 0
        else:
            for i in range(row):
                for j in range(col):
                    if board[i, j] > 0:
                        continue
                    for new_tile in self.tile_spawn_chance:
                        board[i, j] = new_tile
                        tile_score.append(self.maximizer(board, cum_score, depth + 1, start_time, time_limit_ms, max_depth))
                        tile_chance.append(self.tile_spawn_chance[new_tile])
                        board[i, j] = 0

        if not tile_score:
            return cum_score + evaluate_board(board)

        tile_score = np.array(tile_score)
        
        # Đồng bộ biến đếm trống chính xác vào logic phân chia xác suất của nhóm bạn
        if empty_tiles_count > overload_amount:
            tile_chance = np.array(tile_chance) / len(tile_chance)
        else:
            tile_chance = np.array(tile_chance) / (len(tile_chance) / 2)

        initial_mean_score = (tile_score * tile_chance).sum()
        small_score = tile_score <= initial_mean_score

        if sum(small_score) == len(tile_score):
            return initial_mean_score

        tile_chance[small_score] /= riskiness
        total_small_score_chance = tile_chance[small_score].sum()
        tile_chance[~small_score] *= (1 - total_small_score_chance / riskiness) / (1 - total_small_score_chance)

        return float((tile_score * tile_chance).sum())


def get_best_move_expectimax(board: np.ndarray, legal_moves: list[Move], time_budget_ms: float) -> Move:
    """Hàm giao tiếp chính nhận dữ liệu từ agent.py để tìm nước đi tối ưu nhất."""
    if not legal_moves:
        return Move.UP

    start_time = time.time()
    best_move = legal_moves if legal_moves else Move.UP

    empty_count = np.count_nonzero(board == 0)

    # CHIẾN LƯỢC QUẢN LÝ THỜI GIAN NGHIÊM NGẶT THEO SỐ Ô TRỐNG (Phân bổ quỹ 60s hợp lý)
    if empty_count > 9:
        # Bàn cờ rất thoáng (>9 ô trống): Ép Bot đi cực nhanh (tối đa 4ms), nhìn nông để tích lũy thời gian tổng.
        effective_time_limit_ms = min(4.0, time_budget_ms)
        max_depth_limit = 2
    elif empty_count > 5:
        # Giai đoạn thành hình (6-9 ô trống): Giới hạn tối đa 12ms, tìm sâu 3 bước.
        effective_time_limit_ms = min(12.0, time_budget_ms)
        max_depth_limit = 3
    elif empty_count > 2:
        # Giai đoạn trung cuộc căng thẳng (3-5 ô trống): Cho phép dùng tối đa 35ms, tìm sâu 4 bước.
        effective_time_limit_ms = min(35.0, time_budget_ms)
        max_depth_limit = 4
    else:
        # BÀN CỜ CỰC CHẬT (Chỉ còn 1-2 ô trống): Quyết định sống còn, bung hết thời gian còn lại để tìm đường sống.
        effective_time_limit_ms = time_budget_ms
        max_depth_limit = 6

    ai_engine = ExpectimaxExpansion(riskiness=1.0)

    # VÒNG LẶP SÂU DẦN (Iterative Deepening từ depth = 1 -> max_depth_limit)
    for current_max_depth in range(1, max_depth_limit + 1):
        elapsed_ms = (time.time() - start_time) * 1000
        # Chừa hao 1.5ms để kịp ngắt vòng lặp quay đầu an toàn
        if elapsed_ms > (effective_time_limit_ms - 1.5):
            break

        # TỐI ƯU SẮP XẾP NƯỚC ĐI BẰNG CACHE (Move Ordering)
        ordered_moves = list(legal_moves)
        cached_res = lookup_cache(board)
        if cached_res:
            cached_depth, cached_score, cached_move = cached_res
            if cached_move in ordered_moves:
                ordered_moves.remove(cached_move)
                ordered_moves.insert(0, cached_move)

        current_depth_best_move = None
        max_score = float('-inf')

        # DUYỆT TÌM NƯỚC ĐI TỐT NHẤT TRONG TẦNG ĐỘ SÂU HIỆN TẠI
        for move in ordered_moves:
            projected_board, merge_score, valid = move_board(board, move)
            if not valid:
                continue

            score = ai_engine.expectedScore(
                projected_board,
                cum_score=(merge_score * 0.05),
                riskiness=ai_engine.riskiness,
                depth=1,
                start_time=start_time,
                time_limit_ms=effective_time_limit_ms,
                max_depth=current_max_depth
            )

            if score > max_score:
                max_score = score
                current_depth_best_move = move

        # CHỐT KẾT QUẢ TẦNG: Lưu lại vào cache để tối ưu hóa sắp xếp cho các lượt đi tiếp theo
        current_elapsed_ms = (time.time() - start_time) * 1000
        if current_depth_best_move is not None and current_elapsed_ms < effective_time_limit_ms:
            best_move = current_depth_best_move
            store_cache(board, current_max_depth, max_score, best_move)
    return best_move
