"""
Module Bảng Tra Cứu (Lookup Table - LUT) tốc độ cao cho bàn cờ 2048.
Nhiệm vụ của Long: Thay thế hàm move_board Python thuần bằng tra cứu O(1).

Kỹ thuật:
- Mã hóa mỗi hàng 4 ô thành số nguyên 16-bit (4 nibble, mỗi nibble = số mũ log2 của ô)
- Tính toán sẵn kết quả gộp cho TẤT CẢ 65536 hoán vị của 1 hàng
- Lưu vào 4 bảng: LEFT/RIGHT/UP/DOWN (hỗ trợ tra cứu trực tiếp theo chiều)
- Độ phức tạp: O(1) thay vì duyệt vòng lặp Python thuần
"""
from __future__ import annotations

import numpy as np
from game2048.types import Move


# ─── KHỞI TẠO BẢNG LUT ─────────────────────────────────────────────────────

def _encode_row(row) -> int:
    """Mã hóa 1 hàng [a, b, c, d] thành số nguyên 16-bit (4 nibble = số mũ log2)."""
    encoded = 0
    for i, val in enumerate(row):
        if val > 0:
            exp = int(val).bit_length() - 1  # log2 tương đương
            if exp > 15:
                exp = 15
            encoded |= (exp << (i * 4))
    return encoded


def _decode_row(encoded: int) -> np.ndarray:
    """Giải mã số nguyên 16-bit thành hàng [a, b, c, d]."""
    row = np.zeros(4, dtype=np.uint32)
    for i in range(4):
        exp = (encoded >> (i * 4)) & 0xF
        if exp > 0:
            row[i] = 1 << exp
    return row


def _merge_left(row: np.ndarray) -> tuple[np.ndarray, int]:
    """Gộp hàng về bên Trái, trả về (hàng mới, điểm thưởng)."""
    values = [int(v) for v in row if v > 0]
    merged = []
    reward = 0
    i = 0
    while i < len(values):
        if i + 1 < len(values) and values[i] == values[i + 1]:
            val = values[i] * 2
            merged.append(val)
            reward += val
            i += 2
        else:
            merged.append(values[i])
            i += 1
    merged.extend([0] * (4 - len(merged)))
    return np.asarray(merged, dtype=np.uint32), reward


def _build_lut():
    """Xây dựng toàn bộ 4 bảng LUT cho 65536 hoán vị hàng."""
    # Bảng di chuyển: lưu hàng kết quả đã mã hóa
    lut_left  = np.zeros(65536, dtype=np.uint16)
    lut_right = np.zeros(65536, dtype=np.uint16)
    # Bảng điểm thưởng
    lut_left_reward  = np.zeros(65536, dtype=np.uint32)
    lut_right_reward = np.zeros(65536, dtype=np.uint32)

    for enc in range(65536):
        row = _decode_row(enc)

        # Di chuyển Trái
        moved_left, reward_left = _merge_left(row)
        lut_left[enc] = _encode_row(moved_left)
        lut_left_reward[enc] = reward_left

        # Di chuyển Phải = lật hàng → trái → lật lại
        row_flipped = row[::-1]
        moved_right_flipped, reward_right = _merge_left(row_flipped)
        moved_right = moved_right_flipped[::-1]
        lut_right[enc] = _encode_row(moved_right)
        lut_right_reward[enc] = reward_right

    return lut_left, lut_right, lut_left_reward, lut_right_reward


# Khởi tạo LUT một lần duy nhất khi module được import
_LUT_LEFT, _LUT_RIGHT, _LUT_LEFT_REWARD, _LUT_RIGHT_REWARD = _build_lut()


# ─── HÀM ENCODE/DECODE BÀNG CỜ ──────────────────────────────────────────────

def encode_board(board: np.ndarray) -> list[int]:
    """Mã hóa bàn cờ 4x4 thành danh sách 4 số nguyên 16-bit (mỗi phần tử = 1 hàng)."""
    encoded = []
    for row in board:
        encoded.append(_encode_row(row))
    return encoded


def decode_board(encoded_rows: list[int]) -> np.ndarray:
    """Giải mã danh sách 4 hàng mã hóa thành bàn cờ numpy 4x4."""
    rows = [_decode_row(enc) for enc in encoded_rows]
    return np.stack(rows).astype(np.uint32)


# ─── HÀM DI CHUYỂN NHANH ────────────────────────────────────────────────────

def fast_move_board(board: np.ndarray, move: Move) -> tuple[np.ndarray, int, bool]:
    """
    Phiên bản tốc độ cao của move_board sử dụng LUT tra cứu O(1).
    Thay thế hoàn toàn hàm move_board gốc trong vòng lặp Expectimax.
    
    Returns: (new_board, reward, changed)
    """
    if move == Move.LEFT:
        enc_rows = [_encode_row(board[r]) for r in range(4)]
        new_enc = [_LUT_LEFT[e] for e in enc_rows]
        reward = sum(_LUT_LEFT_REWARD[e] for e in enc_rows)
        new_board = decode_board(new_enc)

    elif move == Move.RIGHT:
        enc_rows = [_encode_row(board[r]) for r in range(4)]
        new_enc = [_LUT_RIGHT[e] for e in enc_rows]
        reward = sum(_LUT_RIGHT_REWARD[e] for e in enc_rows)
        new_board = decode_board(new_enc)

    elif move == Move.UP:
        # Transpose → xử lý theo hàng → transpose lại
        t = board.T
        enc_rows = [_encode_row(t[r]) for r in range(4)]
        new_enc = [_LUT_LEFT[e] for e in enc_rows]
        reward = sum(_LUT_LEFT_REWARD[e] for e in enc_rows)
        new_board = decode_board(new_enc).T

    else:  # Move.DOWN
        # Transpose + flip → xử lý theo hàng → flip + transpose lại
        t = board.T
        enc_rows = [_encode_row(t[r]) for r in range(4)]
        new_enc = [_LUT_RIGHT[e] for e in enc_rows]
        reward = sum(_LUT_RIGHT_REWARD[e] for e in enc_rows)
        new_board = decode_board(new_enc).T

    changed = not np.array_equal(new_board, board)
    return new_board, int(reward), changed


def fast_legal_moves(board: np.ndarray) -> list[Move]:
    """Trả về danh sách nước đi hợp lệ, sử dụng fast_move_board."""
    return [move for move in [Move.LEFT, Move.RIGHT, Move.UP, Move.DOWN]
            if fast_move_board(board, move)[2]]
