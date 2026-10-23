"""Nhiệm vụ của Lính cát 2: Quản lý Transposition Table và Zobrist Hashing"""
import numpy as np

# Khởi tạo bảng băm (hash table) toàn cục
transposition_table = {}

def clear_cache():
    """Hàm này được Lead gọi trong hàm reset() để xóa cache giữa các ván"""
    global transposition_table
    transposition_table.clear()

# TODO: Viết thêm các hàm lưu (store) và lấy (lookup) cache