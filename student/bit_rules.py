
from game2048.bitboard import pack_board, unpack_board
from game2048.rules import Move, merge_row_left

import numpy as np

step, strides = 4, 16
def transpose(exponent_bitboard: int) ->int:
    """
    Tranpose the 64-bit board using blocks swapped
    """
    res = exponent_bitboard

    q = (exponent_bitboard^(exponent_bitboard>>24)) & 0xFF00FF00 
    #Allign quarter 2 with quarter 3 on the bitboard, then apply XOR and single them out with mask
    #Representation of t in quarter of 16-bits:
    #   0   |   0
    # Q2^Q3 |   0
    res ^= q^(q<<24)
    #Representation of steps:
    # t<<12 push all of it bit up to from quarter 3 to quarter 2, t^(t<<12) can be shown as
    #   0   | Q2^Q3
    # Q2^Q3 |   0
    #Therfore, when apply the XOR op on it with res, it will be:
    #     Q1     | Q2^(Q2^Q3)  = Q1 | Q3
    # Q3^(Q2^Q3) |     Q4        Q2 | Q4
    
    #Same ideas apply here
    o = ((res>>12)^res) & 0x0000F0F00000F0F0
    res ^= o^(o<<12)
    return res 

def _left_merge_row_with_reward(exponent_bitrow: int)-> tuple[int, int]:
    cells = [(exponent_bitrow>>(4*i)) & 0xF for i in range(4)]
    merged_row = 0
    non_empty = [cell for cell in cells if cell]
    i, reward= 0, 0
    while i<len(non_empty)-1:
        if non_empty[i] == non_empty[i+1] and non_empty[i]<15:
            non_empty.pop(i)
            non_empty[i]+=1
            reward+=(1<<non_empty[i])
        i+=1
    non_empty.reverse()
    #Padding
    merged_row <<= (4-len(non_empty))*4
    for tile in non_empty:
        merged_row = (merged_row<<4) | tile
    return merged_row, reward

def flipleft(exponent_board: int)-> int:
    rows = [(exponent_board>>(16*i)) &0xFFFF for i in range(4)]
    flipped_board = 0
    while rows:
        row = rows.pop()
        cells = [(row>>(4*i)) &0xF for i in range(4)]
        for cell in cells:
            flipped_board = (flipped_board<<4) | cell
    return flipped_board   
        
def move_bitboard(exponent_bitboard: int, move: Move):
    match move:
        case Move.LEFT:
            trans = []
        case Move.RIGHT:
            trans = [flipleft]
        case Move.UP:
            trans = [transpose]
        case Move.DOWN:
            trans = [transpose, flipleft]
    temp_board = exponent_bitboard
    for tran in trans:
        #print(tran.__name__)
        temp_board = tran(temp_board)
        #print(unpack_board(temp_board))
    rows = [(temp_board>>(16*i)) &0xFFFF for i in range(4)]
    res, total_reward = 0, 0
    while rows:
        row = rows.pop()
        total_reward += SCORE[row]
        res = (res<<16) | int(MERGED[row])
    for tran in trans[::-1]:
        res = tran(res)
    return res, total_reward, (res!=exponent_bitboard)

def lookup_table() -> tuple[np.ndarray, np.ndarray]:
    left_merged_result = np.zeros(65536)
    left_merged_score = np.zeros(65536, dtype=np.uint32)

    for row in range(65536):
        merged, score = _left_merge_row_with_reward(row)
        left_merged_result[row] = merged
        left_merged_score[row] = score

    return left_merged_result, left_merged_score

def quick_legal_moves(exponent_bitboard: int) ->list[Move]:
    return [move for move in Move if move_bitboard(exponent_bitboard, move)[2]]

def get_empty(exponent_bitboard: int) -> list[int]:
    return [i for i in range(16) if not ((exponent_bitboard>>(4*i))&0xF)]

def spawn_tile(exponent_bitboard: int, flat_index: int, exponent_tile: int) -> int:
    step = flat_index*4
    target = exponent_bitboard & (0xF<<(step))
    return exponent_bitboard^(target^(exponent_tile<<(step)))

#Unused functions
def _left_merge(exponent_bitboard: int) ->int:
    rows = [(exponent_bitboard>>(16*i)) & 0xFFFF for i in range(4)]
    merged_board = 0
    while rows:
        row = rows.pop()
        cells = [(row>>(4*i)) & 0xF for i in range(4)]
        non_empty = [cell for cell in cells if cell]
        i, reward= 0, 0
        print("Before merged: ", non_empty)
        while i<len(non_empty)-1:
            if non_empty[i] == non_empty[i+1]:
               non_empty.pop(i)
               non_empty[i]+=1
               reward+=(1<<non_empty[i])
               i+=1
            else:
                i+=1
        print("After merged: ", non_empty)
        print("Reward: ", reward)
        non_empty.reverse()
        #Padding
        merged_board <<= (4-len(non_empty))*4
        for tile in non_empty:
            merged_board = (merged_board<<4) | tile
    return merged_board
    
def _left_merge_row(exponent_bitrow: int)-> tuple[int, int]:
    cells = [(exponent_bitrow>>(4*i)) & 0xF for i in range(4)]
    merged_row = 0
    non_empty = [cell for cell in cells if cell]
    i= 0
    while i<len(non_empty)-1:
        if non_empty[i] == non_empty[i+1] and non_empty[i]<16:
            non_empty.pop(i)
            non_empty[i]+=1
            i+=1
        else:
            i+=1
    non_empty.reverse()
    #Padding
    merged_row <<= (4-len(non_empty))*4
    for tile in non_empty:
        merged_row = (merged_row<<4) | tile
    return merged_row   

MERGED, SCORE = lookup_table()


#Testing stuff
if __name__ == "__main__": 
    board = np.array([[2, 0, 2, 4],
                     [4, 8, 16, 0],
                     [0, 32, 8, 16],
                     [4, 16, 128, 8]])
    print(board)
    b_board = transpose(pack_board(board))
    print(unpack_board(b_board))
    print("Left merged:\n",unpack_board(_left_merge(b_board)))
    print(0x00FF.bit_length())
    print("Down merged: \n", unpack_board(move_bitboard(b_board, Move.DOWN)[0]), move_bitboard(b_board, Move.DOWN)[1])
    empty = get_empty(b_board)
    print("Empty: ", empty)
    new_bboard = spawn_tile(b_board, empty[0], 1)
    print(f"Spawn tile 2 at {empty[0]}: \n", unpack_board(new_bboard))
