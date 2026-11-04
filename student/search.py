"""
Nhiệm vụ phối hợp của Linh cát 1 & 2: Viết thuật toán Expectimax, Quản lý thời gian và Cache
Phiên bản tối ưu hóa cấu trúc thế trận và tốc độ thực thi cho Group_6
"""
import time
import random
import numpy as np
import math
from game2048.types import Move
from game2048.rules import legal_moves, is_game_over
from game2048.heuristic import monotonicity, smoothness, merge_potential, HeuristicWeights

# ĐỒNG BỘ LINH CÁT 2: Quản lý bộ nhớ đệm do Phương viết từ file tt_cache.py
from student.tt_cache import lookup_cache, store_cache
# ĐỒNG BỘ LINH CÁT 3: Hàm đánh giá thế cờ dồn góc từ file heuristic.py
from student.heuristic import evaluate_board
# GIAI ĐOẠN 2: Module LUT tốc độ cao - thay thế move_board Python thuần
from student.lut_board import fast_move_board, fast_legal_moves

class ExpectimaxExpansion:
    """Expectimax sử dụng giải thuật đệ quy tích hợp Quản lý thời gian & Cache"""
    def __init__(self, tile_spawn_chance: dict = {2: 0.9, 4: 0.1}, riskiness: float = 1.0):
        self.tile_spawn_chance = tile_spawn_chance
        self.riskiness = riskiness

    def maximizer(self, board: np.ndarray, cum_score: float = 0.0, depth: int = 0, start_time: float = 0.0, time_limit_ms: float = 60000.0, max_depth: int = 4) -> float:
        # BẢO VỆ KHẨN CẤP: Ngắt đệ quy ngay nếu sắp lố thời gian nước đi (chừa hao 2.5ms an toàn)
        if (time.time() - start_time) * 1000 > (time_limit_ms - 2.5):
            return cum_score + evaluate_board(board)

        # ĐIỀU KIỆN DỪNG: Đạt độ sâu tối đa -> Trả về điểm gộp thực tế cộng điểm đánh giá Heuristic thế trận
        if depth >= max_depth:
            return cum_score + evaluate_board(board)

        max_score = float("-inf")
        # Duyệt qua các hướng đi hợp lệ — ưu tiên Trái -> Lên -> Phải -> Xuống
        for move in [Move.LEFT, Move.UP, Move.RIGHT, Move.DOWN]:
            projected_board, merge_score, valid = fast_move_board(board, move)
            if not valid:
                continue
            
            # CÂN BẰNG CHIẾN THUẬT: Thưởng điểm gộp trực tiếp để ưu tiên ghép các ngói lớn
            score = self.expectedScore(projected_board, cum_score + (merge_score * 0.1), self.riskiness, depth + 1, start_time, time_limit_ms, max_depth)
            if score > max_score:
                max_score = score
                
        if max_score == float("-inf"):
            return cum_score + evaluate_board(board)
            
        return max_score

    def expectedScore(self, board: np.ndarray, cum_score: float, riskiness: float, depth: int, start_time: float, time_limit_ms: float = 60000.0, max_depth: int = 4) -> float:
        # BẢO VỆ KHẨN CẤP: Kiểm tra thời gian trong tầng Chance Node (chừa hao 2.5ms)
        if (time.time() - start_time) * 1000 > (time_limit_ms - 2.5):
            return cum_score + evaluate_board(board)
            
        if depth >= max_depth:
            return cum_score + evaluate_board(board)

        tile_score = []
        tile_chance = []
        row, col = board.shape
        overload_amount = 9

        empty_tiles_count = np.count_nonzero(board == 0)

        # GIAI ĐOẠN 3: CẮT TỈA XÁC SUẤT THẤP (PRUNING) ĐỂ ĐẠT DEPTH 5-6
        # Khi bàn cờ thoáng (> 9 ô) hoặc ở độ sâu >= 2 (với > 3 ô trống):
        # Chỉ xét ngói 2 (xác suất 90%), bỏ qua ngói 4 (10%) để giảm số nhánh đi một nửa!
        should_prune_tile4 = (empty_tiles_count > overload_amount) or (depth >= 2 and empty_tiles_count > 3)

        if should_prune_tile4:
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
        
        # Phân chia xác suất tương ứng theo số nhánh đã duyệt
        if should_prune_tile4:
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


def get_best_move_expectimax(board: np.ndarray, legal_moves: list[Move], time_budget_ms: float, agent_instance=None) -> Move:
    """Hàm giao tiếp chính nhận dữ liệu từ agent.py để tìm nước đi tối ưu nhất."""
    if not legal_moves:
        return Move.UP

    start_time = time.time()
    best_move = legal_moves[0] if legal_moves else Move.UP

    # Độ sâu thực tế đạt được - sẽ được truyền ra ngoài qua agent_instance
    reached_depth = 0

    empty_count = np.count_nonzero(board == 0)

    # CHIẾN LƯỢC QUẢN LÝ THỜI GIAN VÀ ĐỘ SÂU THEO SỐ Ô TRỐNG:
    if empty_count > 9:
        # Bàn cờ rất thoáng (>9 ô trống): Đi nhanh (tối đa 6ms), duyệt sâu 2
        effective_time_limit_ms = min(6.0, time_budget_ms)
        max_depth_limit = 2
    elif empty_count > 6:
        # Giai đoạn thành hình (7-9 ô trống): Dùng tối đa 16ms, duyệt sâu 3
        effective_time_limit_ms = min(16.0, time_budget_ms)
        max_depth_limit = 3
    elif empty_count > 3:
        # Giai đoạn trung cuộc (4-6 ô trống): Dùng tối đa 35ms, duyệt sâu 4-5
        effective_time_limit_ms = min(35.0, time_budget_ms)
        max_depth_limit = 5
    else:
        # HẬU CUỘC SỐNG CÒN (1-3 ô trống): Bung toàn bộ ngân sách nước đi, duyệt sâu 5-6!
        effective_time_limit_ms = time_budget_ms
        max_depth_limit = 6

    ai_engine = ExpectimaxExpansion(riskiness=1.0)

    # VÒNG LẶP SÂU DẦN (Iterative Deepening từ depth = 1 -> max_depth_limit)
    for current_max_depth in range(1, max_depth_limit + 1):
        elapsed_ms = (time.time() - start_time) * 1000
        # Chừa hao 2.5ms để kịp ngắt vòng lặp quay đầu an toàn (chỉ ngắt từ depth 2 trở lên)
        if current_max_depth > 1 and elapsed_ms > (effective_time_limit_ms - 2.5):
            break

        # TỐI ƯU SẮP XẾP NƯỚC ĐI BẰNG CACHE & CHIẾN LƯỢC DỒN GÓC (Move Ordering)
        preferred_order = [Move.LEFT, Move.UP, Move.RIGHT, Move.DOWN]
        ordered_moves = [m for m in preferred_order if m in legal_moves]
        cached_res = lookup_cache(board)
        if cached_res:
            cached_depth, cached_score, cached_move = cached_res
            if cached_move in ordered_moves:
                ordered_moves.remove(cached_move)
                ordered_moves.insert(0, cached_move)

        current_depth_best_move = None
        max_score = float('-inf')
        completed_all_moves = True

        # DUYỆT TÌM NƯỚC ĐI TỐT NHẤT TRONG TẦNG ĐỘ SÂU HIỆN TẠI
        # Dùng fast_move_board (LUT O(1)) thay thế move_board Python thuần
        for move in ordered_moves:
            # Ở độ sâu >= 2, kiểm tra thời gian trước khi bắt đầu tính nhánh mới
            if current_max_depth > 1 and (time.time() - start_time) * 1000 > (effective_time_limit_ms - 2.5):
                completed_all_moves = False
                break

            projected_board, merge_score, valid = fast_move_board(board, move)
            if not valid:
                continue

            score = ai_engine.expectedScore(
                projected_board,
                cum_score=(merge_score * 0.1),
                riskiness=ai_engine.riskiness,
                depth=1,
                start_time=start_time,
                time_limit_ms=effective_time_limit_ms,
                max_depth=current_max_depth
            )

            if score > max_score:
                max_score = score
                current_depth_best_move = move

        # CHỐT KẾT QUẢ TẦNG: Cập nhật best_move khi tầng này hoàn thành
        current_elapsed_ms = (time.time() - start_time) * 1000
        if current_depth_best_move is not None:
            if current_max_depth == 1 or (completed_all_moves and current_elapsed_ms < effective_time_limit_ms):
                best_move = current_depth_best_move
                reached_depth = current_max_depth  # Ghi lại độ sâu đã chốt thực tế
                store_cache(board, current_max_depth, max_score, best_move)

    # GIAI ĐOẠN 1: Truyền độ sâu thực tế ra ngoài để benchmark đọc được
    if agent_instance is not None:
        agent_instance.last_completed_depth = reached_depth
        agent_instance.search_depth = reached_depth

    return best_move
