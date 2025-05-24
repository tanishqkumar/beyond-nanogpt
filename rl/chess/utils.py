import chess
import torch 
from typing import List 

TYPES = [chess.PAWN, chess.ROOK, chess.KING,
            chess.QUEEN, chess.BISHOP, chess.KNIGHT]
TYPE_TO_IDX  = {t:i for i,t in enumerate(TYPES)}
COLOR_TO_IDX = {chess.BLACK:0, chess.WHITE:1}

def get_empty(): 
    return chess.Board(fen=None)

def board2input(
        board_history: List[chess.Board]    
    ) -> torch.Tensor: # chess.Board -> [MT+ L, 8, 8] input     

    # Convert single board to list if needed
    if not isinstance(board_history, list):
        board_history = [board_history]

    M = len(board_history)
    if M < 8: 
        padding_boards = [get_empty() for _ in range(8-M)]
        board_history = padding_boards + board_history

    M = len(board_history)
    T = len(TYPES)*2 + 2       # 12 piece‐planes + 2 repetition‐flags
    L = 7                             # turn, ply/1000, castling×4, halfmove/50
    mtl = M*T + L

    out = torch.zeros(mtl, 8, 8, dtype=torch.float32)

    # M * T channels
    for i, board in enumerate(board_history): 
        board_offset = T*i # eg. board 4 means we start indexing ch at 24
        for (pos, piece) in board.piece_map().items(): 
            # map piece to a (ch, r, c)
            t = TYPE_TO_IDX[piece.piece_type]
            c = COLOR_TO_IDX[piece.color]
            piece_offset = t*2 + c # every type channel has two slots, one for each player
            ch = board_offset + piece_offset
            row, col = divmod(pos, 8)
            out[ch, row, col] = 1.

        # hardcode repetition flag channels
        out[board_offset, :, :] = board.is_repetition(2)
        board_offset+=1
        out[board_offset, :, :] = board.is_repetition(4)

    # L channels for special things 
    last = board_history[-1]
    out[board_offset+1, :, :].fill_(int(last.turn))
    out[board_offset+2, :, :].fill_(last.ply()/1000.0)
    out[board_offset+3, :, :].fill_(last.has_kingside_castling_rights(last.turn))
    out[board_offset+4, :, :].fill_(last.has_queenside_castling_rights(last.turn))
    out[board_offset+5, :, :].fill_(last.has_kingside_castling_rights(not last.turn))
    out[board_offset+6, :, :].fill_(last.has_queenside_castling_rights(not last.turn))
    # game is draw if pawns dont move for 50 moves 
    out[board_offset+7, :, :].fill_(last.halfmove_clock / 50.0)

    return out 
