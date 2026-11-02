"""Nhiệm vụ của Lính cát 4: Viết lại hàm move_board để chạy cực nhanh (không dùng copy)"""
import numpy as np
from game2048.types import Move

def get_empty_cells(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == 0))

# TODO: Viết hàm trượt bàn cờ siêu tốc (Fast Move Simulator)
def _slide_left(row: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Hàm phụ: Trượt và gộp một hàng 4 phần tử sang bên trái.
    Trả về (hàng_mới, điểm_cộng).
    """
    # 1. Loại bỏ các số 0 (Nén các ô về bên trái)
    non_zero = row[row != 0]
    
    # 2. Gộp các số giống nhau liền kề (chỉ gộp 1 lần)
    new_row = []
    score = 0
    skip = False
    
    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        # Nếu phần tử hiện tại bằng phần tử kế tiếp -> Gộp
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
            merged_val = non_zero[i] * 2
            new_row.append(merged_val)
            score += merged_val
            skip = True # Đánh dấu đã gộp, bỏ qua phần tử kế tiếp
        else:
            new_row.append(non_zero[i])
            
    # 3. Đệm thêm số 0 vào cuối cho đủ 4 phần tử
    while len(new_row) < 4:
        new_row.append(0)
        
    return np.array(new_row, dtype=row.dtype), score

def fast_move_board(board: np.ndarray, move: Move) -> tuple[np.ndarray, int, bool]:
    """
    Hàm mô phỏng trượt & gộp tối ưu.
    Nhận vào bàn cờ (4x4) và hướng đi, trả về (bàn_cờ_mới, điểm, hợp_lệ).
    """
    # Copy bàn cờ ra một mảng mới để không làm hỏng dữ liệu gốc của thầy
    new_board = board.copy()
    total_score = 0
    
    # Đưa mọi hướng về dạng "Trượt sang Trái" để dễ xử lý
    if move == Move.LEFT:
        pass # Không cần xoay
    elif move == Move.RIGHT:
        new_board = np.fliplr(new_board)
    elif move == Move.UP:
        new_board = np.rot90(new_board, 1) # Xoay 90 độ ngược chiều kim đồng hồ
    elif move == Move.DOWN:
        new_board = np.rot90(new_board, 3) # Xoay 90 độ cùng chiều kim đồng hồ
        
    # Thực hiện trượt trái trên từng hàng
    result_board = np.zeros_like(new_board)
    for i in range(4):
        new_row, row_score = _slide_left(new_board[i, :])
        result_board[i, :] = new_row
        total_score += row_score
        
    # Xoay ngược lại về đúng hướng ban đầu
    if move == Move.LEFT:
        pass
    elif move == Move.RIGHT:
        result_board = np.fliplr(result_board)
    elif move == Move.UP:
        result_board = np.rot90(result_board, 3)
    elif move == Move.DOWN:
        result_board = np.rot90(result_board, 1)
        
    # Kiểm tra xem nước đi có làm thay đổi bàn cờ không (Hợp lệ hay không)
    is_valid = not np.array_equal(board, result_board)
    
    # Ép kiểu trả về cho khớp với hàm của thầy
    return result_board, int(total_score), is_valid


# =====================================================================
#  ĐOẠN CODE TEST
# =====================================================================
if __name__ == "__main__":
    import time
    
    print("🧪 Đang test tính đúng đắn...")
    
    # Test 1: Trượt trái cơ bản (Gộp 1 lần)
    b1 = np.array([
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    nb, score, valid = fast_move_board(b1, Move.LEFT)
    assert np.array_equal(nb[0], [4, 4, 0, 0]), f"Lỗi Test 1: {nb[0]}"
    assert score == 8, f"Lỗi Test 1 Score: {score}"
    assert valid == True
    print("✅ Test 1 (Gộp 2,2,2,2 -> 4,4) Passed!")
    
    # Test 2: Trượt phải
    b2 = np.array([
        [2, 2, 4, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    nb, score, valid = fast_move_board(b2, Move.RIGHT)
    assert np.array_equal(nb[0], [0, 0, 4, 4]), f"Lỗi Test 2: {nb[0]}"
    assert score == 4, f"Lỗi Test 2 Score: {score}"
    print("✅ Test 2 (Trượt phải) Passed!")
    
    # Test 3: Trượt xuống
    b3 = np.array([
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0]
    ])
    nb, score, valid = fast_move_board(b3, Move.DOWN)
    assert np.array_equal(nb[:, 0], [0, 0, 4, 8]), f"Lỗi Test 3: {nb[:, 0]}"
    assert score == 12, f"Lỗi Test 3 Score: {score}"
    print("✅ Test 3 (Trượt xuống) Passed!")

    # Test Tốc độ (Chạy 100,000 lần để xem mất bao nhiêu giây)
    print("\n⚡ Đang test tốc độ (100,000 lần mô phỏng)...")
    board_random = np.random.choice([0, 2, 4, 8, 16], size=(4, 4))
    start = time.time()
    for _ in range(100000):
        fast_move_board(board_random, Move.LEFT)
    end = time.time()
    print(f"⏱️ Thời gian chạy 100,000 nước đi: {end - start:.3f} giây")
    print("👉 (Nếu dưới 2 giây là code của mày đã đủ nhanh để Expectimax chạy ầm ầm)")