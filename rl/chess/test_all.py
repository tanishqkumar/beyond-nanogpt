'''
Run tests from rl/chess with `pytest test_all.py -v` 
'''
import chess 
import torch
from model import ChessNet 
from utils import move2index
from env import ChessEnv
import pytest 


## define objects/constants we'll use in tests in the way pytests wants ##
@pytest.fixture
def net(): 
    return ChessNet(in_ch=119)

@pytest.fixture
def board(): 
    return chess.Board()

@pytest.fixture
def nactions(): 
    return 4672

@pytest.fixture
def env(): 
    return ChessEnv()

## define tests, they will run by themselves when pytest calls this file 

def test_forward_pass(net: ChessNet, board: chess.Board, nactions: int):
    history = [board]
    
    v, l = net(history)

    assert v.shape == (1, 1) and l.shape == (1, nactions)

def test_move_masking(net: ChessNet, board: chess.Board):
    legal_move = chess.Move.from_uci("e2e4")
    illegal_move = chess.Move.from_uci("e2e6")
    
    _, logits = net(board)
    
    legal_idx = move2index(board, legal_move)
    illegal_idx = move2index(board, illegal_move)
    
    # Verify legality is correctly masked 
    assert torch.isinf(logits[0, illegal_idx]) and logits[0, illegal_idx] < 0, "Illegal move should be masked with -inf"
    assert not torch.isinf(logits[0, legal_idx]), "Legal move should not be masked"

def test_env_clone(env: ChessEnv): 
    # Make a move to change the state
    legal_moves = list(env.board.legal_moves)
    move_idx = move2index(env.board, legal_moves[0])
    env.step(move_idx)
    
    # Clone the environment
    cloned_env = env.clone()
    
    # Verify the clone has the same state
    assert cloned_env.move_count == env.move_count
    assert cloned_env.board.fen() == env.board.fen()
    assert cloned_env.max_moves == env.max_moves
    assert cloned_env.history_len == env.history_len
    assert len(cloned_env.board_history) == len(env.board_history)
    
    # Verify they are independent (modifying one doesn't affect the other)
    if list(cloned_env.board.legal_moves):
        next_move_idx = move2index(cloned_env.board, list(cloned_env.board.legal_moves)[0])
        cloned_env.step(next_move_idx)
        assert cloned_env.move_count != env.move_count

def test_env_reset(env: ChessEnv): 
    # Make some moves to change the state
    legal_moves = list(env.board.legal_moves)
    move_idx = move2index(env.board, legal_moves[0])
    env.step(move_idx)
    
    original_move_count = env.move_count
    original_board_fen = env.board.fen()
    
    # Reset the environment
    env.reset()
    
    # Verify reset worked
    assert env.move_count == 0
    assert env.board.fen() == chess.Board().fen()  # Should be starting position
    assert len(env.board_history) == 1  # Should only have initial board
    assert env.move_count != original_move_count
    assert env.board.fen() != original_board_fen

def test_env_step(env: ChessEnv): 
    initial_move_count = env.move_count
    initial_fen = env.board.fen()
    
    # Get a legal move
    legal_moves = list(env.board.legal_moves)
    assert len(legal_moves) > 0, "Should have legal moves in starting position"
    
    move = legal_moves[0]
    move_idx = move2index(env.board, move)
    
    # Take a step
    s_next, reward, done, info = env.step(move_idx)
    
    # Verify the step worked
    assert env.move_count == initial_move_count + 1
    assert env.board.fen() != initial_fen
    assert s_next.shape[0] == 119  # Should return proper state tensor
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert len(env.board_history) == min(env.move_count + 1, env.history_len)
    
    # Test illegal move raises exception
    with pytest.raises(Exception, match="illegal move"):
        # Try to make the same move again (should be illegal)
        same_move_new_idx = move2index(env.board, move)
        env.step(same_move_new_idx)
