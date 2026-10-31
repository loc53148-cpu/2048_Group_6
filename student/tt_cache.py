"""Nhiệm vụ của Lính cát 2: Quản lý Transposition Table và Zobrist Hashing"""
import numpy as np

# Khởi tạo bảng băm (hash table) toàn cục dưới dạng Dictionary của Python
transposition_table = {}

# Cài đặt ma trận Zobrist Table cố định bằng một seed để bảo đảm mã băm đồng nhất suốt ván đấu
# Bàn cờ 16 ô, mỗi ô chứa tối đa cấu hình lũy thừa từ 0 đến 15 (Tile trống cho đến tile 32768)
_rng = np.random.default_rng(42)
_ZOBRIST_MATRIX = _rng.integers(0, 2**63, size=(16, 16), dtype=np.uint64)

def _get_zobrist_hash(board: np.ndarray) -> np.uint64:
    """Hàm băm ma trận bàn cờ 4x4 thành mã số nguyên 64-bit duy nhất bằng phép toán XOR."""
    flat_board = board.flatten()
    hash_key = np.uint64(0)
    
    for i in range(16):
        val = flat_board[i]
        # Chuyển đổi giá trị ô số sang số mũ cơ số 2 (0->0, 2->1, 4->2, 8->3,...)
        power = int(np.log2(val)) if val > 0 else 0
        if power > 15:
            power = 15
        hash_key ^= _ZOBRIST_MATRIX[i][power]
        
    return hash_key

def clear_cache():
    """Hàm này được Lead gọi trong hàm reset() để xóa sạch cache khi bắt đầu ván đấu mới"""
    global transposition_table
    transposition_table.clear()

def lookup_cache(board: np.ndarray):
    """
    Tra cứu bàn cờ hiện tại trong bảng hoán vị.
    Trả về: (depth, score, best_move) nếu tìm thấy, ngược lại trả về None.
    """
    hash_key = _get_zobrist_hash(board)
    return transposition_table.get(hash_key, None)

def store_cache(board: np.ndarray, depth: int, score: float, best_move):
    """
    Lưu thông tin phân tích của thế cờ vào bảng hoán vị.
    Chỉ lưu hoặc ghi đè nếu thế cờ này chưa có hoặc lượt này tìm kiếm sâu hơn lượt cũ.
    """
    hash_key = _get_zobrist_hash(board)
    if hash_key not in transposition_table or depth > transposition_table[hash_key][0]:
        transposition_table[hash_key] = (depth, score, best_move)
