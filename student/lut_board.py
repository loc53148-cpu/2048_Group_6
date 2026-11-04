"""
Module Bảng Tra Cứu (Lookup Table - LUT) siêu tốc cho bàn cờ 2048.
Nhiệm vụ của Long: Tối ưu hóa tối đa, loại bỏ hoàn toàn các overhead tạo mảng trung gian.

Kỹ thuật:
- Tính toán sẵn bảng tra cứu cho tất cả các hoán vị của 1 hàng (từ [0,0,0,0] đến 65536)
- Lưu trữ dưới dạng Tuple Hash Table trực tiếp: O(1) C-speed lookup
- Kiểm tra tính hợp lệ (valid/changed) bằng phép so sánh Tuple cực nhanh
- Nếu nước đi không hợp lệ: Trả về False ngay lập tức, KHÔNG sinh mảng mới
- Xử lý UP/DOWN trực tiếp qua cột mà không cần Transpose (.T) hay Fliplr
"""
from __future__ import annotations

import numpy as np
from game2048.types import Move


def _merge_tuple_left(row: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], int]:
    """Hàm gộp 1 hàng 4 phần tử về bên trái, trả về (hàng_mới, điểm_thưởng)."""
    vals = [v for v in row if v != 0]
    res = []
    reward = 0
    i = 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            val = vals[i] * 2
            res.append(val)
            reward += val
            i += 2
        else:
            res.append(vals[i])
            i += 1
    while len(res) < 4:
        res.append(0)
    return (res[0], res[1], res[2], res[3]), reward


def _build_lut():
    """Khởi tạo trước toàn bộ bảng tra cứu cho mọi giá trị ngói có thể xuất hiện."""
    tiles = [0] + [1 << i for i in range(1, 17)]  # 0, 2, 4, ..., 65536
    lut_left = {}
    lut_right = {}
    for a in tiles:
        for b in tiles:
            for c in tiles:
                for d in tiles:
                    row = (a, b, c, d)
                    # Di chuyển Trái
                    m_left, rew_left = _merge_tuple_left(row)
                    lut_left[row] = (m_left, rew_left)

                    # Di chuyển Phải = lật ngược -> gộp trái -> lật lại
                    row_rev = (d, c, b, a)
                    m_rev, rew_right = _merge_tuple_left(row_rev)
                    m_right = (m_rev[3], m_rev[2], m_rev[1], m_rev[0])
                    lut_right[row] = (m_right, rew_right)
    return lut_left, lut_right


# Khởi tạo bảng một lần duy nhất khi import module
_LUT_LEFT, _LUT_RIGHT = _build_lut()


def fast_move_board(board: np.ndarray, move: Move) -> tuple[np.ndarray, int, bool]:
    """
    Di chuyển bàn cờ bằng LUT O(1), tốc độ vượt trội so với move_board gốc.
    Returns: (new_board, reward, changed)
    """
    b = board
    try:
        if move == Move.LEFT:
            r0 = (int(b[0, 0]), int(b[0, 1]), int(b[0, 2]), int(b[0, 3]))
            r1 = (int(b[1, 0]), int(b[1, 1]), int(b[1, 2]), int(b[1, 3]))
            r2 = (int(b[2, 0]), int(b[2, 1]), int(b[2, 2]), int(b[2, 3]))
            r3 = (int(b[3, 0]), int(b[3, 1]), int(b[3, 2]), int(b[3, 3]))

            m0, rew0 = _LUT_LEFT[r0]
            m1, rew1 = _LUT_LEFT[r1]
            m2, rew2 = _LUT_LEFT[r2]
            m3, rew3 = _LUT_LEFT[r3]

            if m0 == r0 and m1 == r1 and m2 == r2 and m3 == r3:
                return board, 0, False

            new_board = np.array([m0, m1, m2, m3], dtype=np.uint32)
            return new_board, rew0 + rew1 + rew2 + rew3, True

        elif move == Move.RIGHT:
            r0 = (int(b[0, 0]), int(b[0, 1]), int(b[0, 2]), int(b[0, 3]))
            r1 = (int(b[1, 0]), int(b[1, 1]), int(b[1, 2]), int(b[1, 3]))
            r2 = (int(b[2, 0]), int(b[2, 1]), int(b[2, 2]), int(b[2, 3]))
            r3 = (int(b[3, 0]), int(b[3, 1]), int(b[3, 2]), int(b[3, 3]))

            m0, rew0 = _LUT_RIGHT[r0]
            m1, rew1 = _LUT_RIGHT[r1]
            m2, rew2 = _LUT_RIGHT[r2]
            m3, rew3 = _LUT_RIGHT[r3]

            if m0 == r0 and m1 == r1 and m2 == r2 and m3 == r3:
                return board, 0, False

            new_board = np.array([m0, m1, m2, m3], dtype=np.uint32)
            return new_board, rew0 + rew1 + rew2 + rew3, True

        elif move == Move.UP:
            c0 = (int(b[0, 0]), int(b[1, 0]), int(b[2, 0]), int(b[3, 0]))
            c1 = (int(b[0, 1]), int(b[1, 1]), int(b[2, 1]), int(b[3, 1]))
            c2 = (int(b[0, 2]), int(b[1, 2]), int(b[2, 2]), int(b[3, 2]))
            c3 = (int(b[0, 3]), int(b[1, 3]), int(b[2, 3]), int(b[3, 3]))

            m0, rew0 = _LUT_LEFT[c0]
            m1, rew1 = _LUT_LEFT[c1]
            m2, rew2 = _LUT_LEFT[c2]
            m3, rew3 = _LUT_LEFT[c3]

            if m0 == c0 and m1 == c1 and m2 == c2 and m3 == c3:
                return board, 0, False

            new_board = np.array([
                [m0[0], m1[0], m2[0], m3[0]],
                [m0[1], m1[1], m2[1], m3[1]],
                [m0[2], m1[2], m2[2], m3[2]],
                [m0[3], m1[3], m2[3], m3[3]],
            ], dtype=np.uint32)
            return new_board, rew0 + rew1 + rew2 + rew3, True

        else:  # Move.DOWN
            c0 = (int(b[0, 0]), int(b[1, 0]), int(b[2, 0]), int(b[3, 0]))
            c1 = (int(b[0, 1]), int(b[1, 1]), int(b[2, 1]), int(b[3, 1]))
            c2 = (int(b[0, 2]), int(b[1, 2]), int(b[2, 2]), int(b[3, 2]))
            c3 = (int(b[0, 3]), int(b[1, 3]), int(b[2, 3]), int(b[3, 3]))

            m0, rew0 = _LUT_RIGHT[c0]
            m1, rew1 = _LUT_RIGHT[c1]
            m2, rew2 = _LUT_RIGHT[c2]
            m3, rew3 = _LUT_RIGHT[c3]

            if m0 == c0 and m1 == c1 and m2 == c2 and m3 == c3:
                return board, 0, False

            new_board = np.array([
                [m0[0], m1[0], m2[0], m3[0]],
                [m0[1], m1[1], m2[1], m3[1]],
                [m0[2], m1[2], m2[2], m3[2]],
                [m0[3], m1[3], m2[3], m3[3]],
            ], dtype=np.uint32)
            return new_board, rew0 + rew1 + rew2 + rew3, True

    except KeyError:
        # Cơ chế an toàn phòng hờ ngói siêu lớn (> 65536)
        from game2048.rules import move_board
        return move_board(board, move)


def fast_legal_moves(board: np.ndarray) -> list[Move]:
    """Trả về danh sách nước đi hợp lệ nhanh chóng."""
    return [move for move in [Move.LEFT, Move.RIGHT, Move.UP, Move.DOWN]
            if fast_move_board(board, move)[2]]
