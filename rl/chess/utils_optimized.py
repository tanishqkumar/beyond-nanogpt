import chess
import torch 
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING, Union
from collections import deque
import numpy as np
if TYPE_CHECKING: 
    from MCTS import Node 

TYPES = [chess.PAWN, chess.ROOK, chess.KING,
            chess.QUEEN, chess.BISHOP, chess.KNIGHT]
TYPE_TO_IDX  = {t:i for i,t in enumerate(TYPES)}
COLOR_TO_IDX = {chess.BLACK:0, chess.WHITE:1}
_EMPTY = chess.Board(fen=None)

# Precompute position mappings for faster indexing
_POS_TO_ROWCOL = np.array([(pos // 8, pos % 8) for pos in range(64)], dtype=np.int32)
_ROWCOL_TO_POS = np.arange(64, dtype=np.int32).reshape(8, 8)

# Cache for empty board tensor representations
_EMPTY_BOARD_CACHE = {}

def get_empty(): 
    return _EMPTY # this is OK since these are never mutated


def _get_empty_board_tensor(device: torch.device) -> torch.Tensor:
    """Get a cached empty board tensor for the given device."""
    if device not in _EMPTY_BOARD_CACHE:
        _EMPTY_BOARD_CACHE[device] = torch.zeros(14, 8, 8, dtype=torch.float32, device=device)
    return _EMPTY_BOARD_CACHE[device].clone()


def board2input(
        board_history: Union[deque[chess.Board], chess.Board],
        device: torch.device = torch.device("cuda")    
    ) -> torch.Tensor: # chess.Board -> [MT+ L, 8, 8] input     

    # Convert single board to deque if needed
    if isinstance(board_history, chess.Board):
        dq = deque(maxlen=8)
        dq.append(board_history)
        board_history = dq

    M = len(board_history)
    
    # Pad history if needed - reuse the same empty board reference
    if M < 8:
        empty_board = get_empty()
        padded_history = deque(maxlen=8)
        # Add empty boards for padding
        for _ in range(8-M):
            padded_history.append(empty_board)
        # Add actual boards
        padded_history.extend(board_history)
        board_history = padded_history
        M = 8

    T = 14  # 12 piece-planes + 2 repetition-flags
    L = 7   # turn, ply/1000, castling×4, halfmove/50
    mtl = M*T + L

    # Allocate output tensor once
    out = torch.zeros(mtl, 8, 8, dtype=torch.float32, device=device)
    
    # Process boards more efficiently
    for i, board in enumerate(board_history):
        if board is _EMPTY:
            # Skip empty boards - they're already zeros
            continue
            
        board_offset = T * i
        
        # Batch process all pieces using vectorized operations
        piece_map = board.piece_map()
        if piece_map:  # Only process if there are pieces
            # Pre-allocate arrays for batch processing
            positions = list(piece_map.keys())
            pieces = list(piece_map.values())
            
            # Vectorized position calculation
            rows = np.array([pos // 8 for pos in positions], dtype=np.int32)
            cols = np.array([pos % 8 for pos in positions], dtype=np.int32)
            
            # Process each piece
            for idx, (pos, piece) in enumerate(zip(positions, pieces)):
                t = TYPE_TO_IDX[piece.piece_type]
                c = COLOR_TO_IDX[piece.color]
                ch = board_offset + t*2 + c
                out[ch, rows[idx], cols[idx]] = 1.

        # Repetition flags - use fill_ for efficiency
        if board.is_repetition(2):
            out[board_offset + T - 2].fill_(1.)
        if board.is_repetition(4):
            out[board_offset + T - 1].fill_(1.)

    # Special channels - use the last board
    last = board_history[-1]
    base_idx = M * T
    
    # Use in-place operations for better performance
    out[base_idx].fill_(float(last.turn))
    out[base_idx + 1].fill_(last.ply() / 1000.0)
    out[base_idx + 2].fill_(float(last.has_kingside_castling_rights(last.turn)))
    out[base_idx + 3].fill_(float(last.has_queenside_castling_rights(last.turn)))
    out[base_idx + 4].fill_(float(last.has_kingside_castling_rights(not last.turn)))
    out[base_idx + 5].fill_(float(last.has_queenside_castling_rights(not last.turn)))
    out[base_idx + 6].fill_(last.halfmove_clock / 50.0)

    return out


# Cache the template dictionary and its inverse
_TEMPLATE_DICT_CACHE = None
_INV_TEMPLATE_DICT_CACHE = None

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


def get_inv_template_dict() -> Dict[int, Tuple[int, int, Optional[chess.PieceType]]]:
    """Get the inverse template dictionary, cached."""
    global _INV_TEMPLATE_DICT_CACHE
    if _INV_TEMPLATE_DICT_CACHE is not None:
        return _INV_TEMPLATE_DICT_CACHE
    
    template_dict = get_template_dict()
    _INV_TEMPLATE_DICT_CACHE = {v: k for k, v in template_dict.items()}
    return _INV_TEMPLATE_DICT_CACHE


# Optimized move2index with minimal operations
def move2index(board: chess.Board, move: chess.Move) -> int: 
    from_sq, to_sq = move.from_square, move.to_square
    
    # Conditional inversion for black
    if board.turn == chess.BLACK: 
        from_sq = 63 - from_sq
        to_sq = 63 - to_sq
    
    # Efficient rank/file extraction using bit operations
    from_rank = from_sq >> 3  # Equivalent to // 8
    to_rank = to_sq >> 3
    from_file = from_sq & 7   # Equivalent to % 8
    to_file = to_sq & 7
    
    # Calculate deltas
    dx = to_rank - from_rank
    dy = to_file - from_file
    
    # Get template ID
    promo = None if move.promotion == chess.QUEEN else move.promotion
    k = get_template_dict()[(dx, dy, promo)]
    
    return (k << 6) + from_sq  # Equivalent to 64 * k + from_sq


# Optimized index2move
def index2move(board: chess.Board, action: int) -> chess.Move: 
    k = action >> 6  # Equivalent to // 64
    from_sq = action & 63  # Equivalent to % 64

    # Use cached inverse template dict
    dx, dy, promo = get_inv_template_dict()[k]
    to_sq = from_sq + (dx << 3) + dy  # dx * 8 + dy

    # Conditional inversion for black
    if board.turn == chess.BLACK:
        from_sq = 63 - from_sq
        to_sq = 63 - to_sq

    # Check for pawn promotion
    if promo is None and board.piece_type_at(from_sq) == chess.PAWN:
        to_rank = to_sq >> 3
        if to_rank == 0 or to_rank == 7:
            promo = chess.QUEEN

    return chess.Move(from_sq, to_sq, promo)


# Optimized legal_mask using pre-allocated tensors
def legal_mask(logits: torch.Tensor, nodes: List['Node'], nactions: int = 4672) -> torch.Tensor: 
    assert logits.dim() == 2
    b, _ = logits.shape
    device = logits.device
    
    # Use clone() instead of creating new tensor and filling
    result = logits.clone()
    
    # Batch process all nodes at once if possible
    if b == 1:
        # Common case: single node
        legal_indices = nodes[0].legal_indices
        if legal_indices:
            # Create mask for illegal moves
            mask = torch.ones(nactions, dtype=torch.bool, device=device)
            mask[legal_indices] = False
            result[0, mask] = float('-inf')
    else:
        # Multiple nodes: process in batch
        for i in range(b):
            legal_indices = nodes[i].legal_indices
            if legal_indices:
                # Create mask for illegal moves
                mask = torch.ones(nactions, dtype=torch.bool, device=device)
                mask[legal_indices] = False
                result[i, mask] = float('-inf')
    
    return result


# Cache for terminal position evaluations to avoid repeated outcome checks
_EVAL_CACHE = {}

def eval_pos(board: chess.Board) -> Tuple[float, bool, Dict[str, Any]]: 
    # Try to use cached result if available
    board_hash = board._transposition_key()
    if board_hash in _EVAL_CACHE:
        return _EVAL_CACHE[board_hash]
    
    info = {}
    
    if not board.is_game_over():
        result = (0., False, info)
        _EVAL_CACHE[board_hash] = result
        return result
    
    # Game is over - determine outcome
    outcome = board.outcome()
    info["game_over"] = True
    info["fen"] = board.fen()
    
    # Handle checkmate
    if outcome.termination == chess.Termination.CHECKMATE:
        if outcome.winner == board.turn:
            info["termination_reason"] = "checkmate"
            info["winner"] = board.turn
            result = (-1., True, info)
        else:
            info["termination_reason"] = "checkmate"
            info["winner"] = not board.turn
            result = (1., True, info)
    else:
        # Handle all draw conditions
        termination_map = {
            chess.Termination.STALEMATE: "stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
            chess.Termination.SEVENTYFIVE_MOVES: "seventyfive_moves",
            chess.Termination.FIVEFOLD_REPETITION: "fivefold_repetition",
            chess.Termination.FIFTY_MOVES: "fifty_moves",
            chess.Termination.THREEFOLD_REPETITION: "threefold_repetition"
        }
        info["termination_reason"] = termination_map.get(outcome.termination, "unknown")
        info["winner"] = None  # Draw
        result = (0., True, info)
    
    # Cache the result
    if len(_EVAL_CACHE) < 10000:  # Limit cache size
        _EVAL_CACHE[board_hash] = result
    
    return result