import chess
import torch 
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from collections import deque
if TYPE_CHECKING: 
    from MCTS import Node 

TYPES = [chess.PAWN, chess.ROOK, chess.KING,
            chess.QUEEN, chess.BISHOP, chess.KNIGHT]
TYPE_TO_IDX  = {t:i for i,t in enumerate(TYPES)}
COLOR_TO_IDX = {chess.BLACK:0, chess.WHITE:1}
_EMPTY = chess.Board(fen=None)


def get_empty(): 
    return _EMPTY # this is OK since these are never mutated


def board2input(
        board_history: deque[chess.Board],
        device: torch.device = torch.device("cuda")    
    ) -> torch.Tensor: # chess.Board -> [MT+ L, 8, 8] input     

    # check ideal speedup -- THIS IS STILL SLOW, SO NOT HTE BOTTLENECK
    # return torch.zeros(119, 8, 8, dtype=torch.float32, device=device)

    # Convert single board to list if needed
    if isinstance(board_history, chess.Board):
        dq = deque(maxlen=8)
        dq.append(board_history)
        board_history = dq

    M = len(board_history)
    if M < 8: 
        padding_boards = [get_empty() for _ in range(8-M)]
        padded_history = deque(maxlen=8)
        for board in padding_boards:
            padded_history.append(board)
        for board in board_history:
            padded_history.append(board)
        board_history = padded_history

    M = len(board_history)
    T = len(TYPES)*2 + 2       # 12 piece‐planes + 2 repetition‐flags
    L = 7                             # turn, ply/1000, castling×4, halfmove/50
    mtl = M*T + L

    out = torch.zeros(mtl, 8, 8, dtype=torch.float32, device=device)

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

        # hardcode repetition flag channels as last 2 in every set of T=14
        out[board_offset+(T-2), :, :] = board.is_repetition(2)
        out[board_offset+(T-1), :, :] = board.is_repetition(4)

    # L channels for special things 
    last = board_history[-1]
    out[M*T, :, :].fill_(int(last.turn))
    out[M*T+1, :, :].fill_(last.ply()/1000.0)
    out[M*T+2, :, :].fill_(last.has_kingside_castling_rights(last.turn))
    out[M*T+3, :, :].fill_(last.has_queenside_castling_rights(last.turn))
    out[M*T+4, :, :].fill_(last.has_kingside_castling_rights(not last.turn))
    out[M*T+5, :, :].fill_(last.has_queenside_castling_rights(not last.turn))
    # game is draw if pawns dont move for 50 moves 
    out[M*T+6, :, :].fill_(last.halfmove_clock / 50.0)

    return out 

# The first 56 planes encode
# possible 'queen moves' for any piece: a number of squares [1..7] in which the piece will be
# moved, along one of eight relative compass directions {N,NE,E,SE,S,SW,W,NW} * len

# output in 0-72 template planes 
# Cache the template dictionary since it's expensive to compute and never changes
_TEMPLATE_DICT_CACHE = None

def get_template_dict() -> Dict[Tuple[int, int, Optional[chess.PieceType]], int]:
    """Return a dict mapping all possible (dx, dy, promo) tuples to template IDs."""
    
    global _TEMPLATE_DICT_CACHE
    if _TEMPLATE_DICT_CACHE is not None:
        return _TEMPLATE_DICT_CACHE
    
    template_dict = {}
    
    # Queen-like moves (0-55): 8 directions × 7 distances
    directions = [
        (0, 1),   # N
        (1, 1),   # NE  
        (1, 0),   # E
        (1, -1),  # SE
        (0, -1),  # S
        (-1, -1), # SW
        (-1, 0),  # W
        (-1, 1)   # NW
    ]
    
    for dir_idx, (dx_unit, dy_unit) in enumerate(directions):
        for distance in range(1, 8):  # 1-7 squares
            dx = dx_unit * distance
            dy = dy_unit * distance
            template_id = dir_idx * 7 + (distance - 1)
            template_dict[(dx, dy, None)] = template_id
    
    # Knight moves (56-63): 8 possible knight moves
    knight_moves = [
        (1, 2),   # NNE
        (2, 1),   # ENE
        (2, -1),  # ESE
        (1, -2),  # SSE
        (-1, -2), # SSW
        (-2, -1), # WSW
        (-2, 1),  # WNW
        (-1, 2)   # NNW
    ]
    
    for knight_idx, (dx, dy) in enumerate(knight_moves):
        template_dict[(dx, dy, None)] = 56 + knight_idx
    
    # Underpromotion moves (64-72): 3 piece types × 3 directions
    promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    promotion_directions = [
        (1, -1),  # diagonal left
        (1, 1),   # diagonal right  
        (1, 0)    # straight ahead
    ]
    
    for piece_idx, piece_type in enumerate(promotion_pieces):
        for dir_idx, (dx, dy) in enumerate(promotion_directions):
            template_id = 64 + piece_idx * 3 + dir_idx
            template_dict[(dx, dy, piece_type)] = template_id
    
    _TEMPLATE_DICT_CACHE = template_dict
    return template_dict

# takes eg. "b5e7" -> idx 2373 out of 4672 in action idx 
def move2index(board: chess.Board, move: chess.Move) -> int: 
    from_sq, to_sq = move.from_square, move.to_square
    
    # invert if black bc this should always be from player's pov
    if board.turn == chess.BLACK: 
        from_sq, to_sq = 63 - from_sq, 63 - to_sq
    
    from_rank, to_rank = from_sq//8, to_sq//8
    from_file, to_file = from_sq % 8, to_sq % 8
    # get dx, dy
    dx, dy = to_rank - from_rank, to_file - from_file
    # get channel (template id)
    dct = get_template_dict()
    # fold queen promotion into regular queen like move 
    promo = None if move.promotion == chess.QUEEN else move.promotion
    k = dct[(dx, dy, promo)]
    
    return 64 * k + from_sq


# define the inverse, for use in eg. chess env to take (s, a) -> (s', r, d) 
    #   when parsing action, a, an int in [nactions], to board.move(a), a chess.Move
def index2move(board: chess.Board, action: int) -> chess.Move: 
    k, from_sq = divmod(action, 64)

    # use template id to see if it involves promotion
    template_dct = get_template_dict()
    inv_template_dct = {v:k for k,v in template_dct.items()}

    dx, dy, promo = inv_template_dct[k]
    to_sq = from_sq + dx * 8 + dy

    # invert to_sq if black, it's important this is before promo logic because board.piece_type_at
        # assumes from_sq is the real (ie. white/black) sq not just from white's pov like board.step 
    if board.turn == chess.BLACK:
        from_sq = 63 - from_sq
        to_sq = 63 - to_sq

    # this was missing a one-rank check before, leading to invalid promotions being registered
    if promo is None and board.piece_type_at(from_sq) == chess.PAWN:
        actual_dx = (to_sq // 8) - (from_sq // 8)
        # must be a one-rank step and land on rank 0 or 7
        if abs(actual_dx) == 1 and (to_sq // 8 == 0 or to_sq // 8 == 7):
            promo = chess.QUEEN

    return chess.Move(from_sq, to_sq, promo)


# takes a board to possible next moves that are legal out of [4672] moves
def legal_mask(logits: torch.Tensor, nodes: List['Node'], nactions: int = 4672) -> torch.Tensor: 
    assert logits.dim() == 2
    b, _ = logits.shape
    device = logits.device  
    mask = torch.zeros((b, nactions), dtype=torch.bool, device=device)
    
    for i in range(b):
        indices = torch.tensor(nodes[i].legal_indices, dtype=torch.long, device=device)
        mask[i, indices] = True

    return logits.masked_fill(~mask, float('-inf'))


def eval_pos(board: chess.Board) -> Tuple[float, bool, Dict[str, Any]]: 
        info = {}
        if board.is_game_over(): 
            # casework on cause
            outcome = board.outcome()
            if outcome.termination == chess.Termination.CHECKMATE and outcome.winner == board.turn: 
                info["game_over"] = True
                info["termination_reason"] = "checkmate"
                info["winner"] = board.turn 
                info["fen"] = board.fen()
                
                return (-1., True, info)
            elif outcome.termination == chess.Termination.CHECKMATE and outcome.winner != board.turn:
                info["game_over"] = True
                info["termination_reason"] = "checkmate"
                info["winner"] = not board.turn 
                info["fen"] = board.fen()
                
                return (1., True, info)
            else:
                # Handle draws, stalemates, and other game endings
                info["game_over"] = True
                info["fen"] = board.fen()
                
                if outcome.termination == chess.Termination.STALEMATE:
                    info["termination_reason"] = "stalemate"
                elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
                    info["termination_reason"] = "insufficient_material"
                elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                    info["termination_reason"] = "seventyfive_moves"
                elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                    info["termination_reason"] = "fivefold_repetition"
                elif outcome.termination == chess.Termination.FIFTY_MOVES:
                    info["termination_reason"] = "fifty_moves"
                elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
                    info["termination_reason"] = "threefold_repetition"
                else:
                    info["termination_reason"] = "unknown"
                
                info["winner"] = None  # Draw
                return (0., True, info)
        else: 
            return (0., False, info) # no reward, game not over
