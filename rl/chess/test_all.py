'''
Run tests from rl/chess with `pytest test_all.py -v` 
'''
import chess 
import torch
from model import ChessNet 
from utils import move2index
from env import ChessEnv
from buffer import Buffer
from MCTS import MCTS, get_root
from config import MCTSConfig, ModelConfig
from collections import deque
import pytest 


## define objects/constants we'll use in tests in the way pytests wants ##
@pytest.fixture
def net(): 
    cfg = ModelConfig()
    return ChessNet(cfg)

@pytest.fixture
def board(): 
    return chess.Board()

@pytest.fixture
def nactions(): 
    return 4672

@pytest.fixture
def env(): 
    return ChessEnv()

@pytest.fixture
def buffer(): 
    return Buffer(buff_sz = 10_000)

## define tests, they will run by themselves when pytest calls this file 

def test_forward_pass(net: ChessNet, board: chess.Board, nactions: int):
    history = deque([board], maxlen=8)
    
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

def test_buffer(buffer: Buffer, board: chess.Board, net: ChessNet, nactions: int): # test push and get_batch
    # Test pushing to buffer
    history = deque([board], maxlen=8)
    val, policy = net(history)
    val_scalar = val.item()
    action = torch.randint(0, nactions, (nactions,))  # Create action tensor with proper shape
    buffer.push(board, val_scalar, action)
    
    # Push a few more samples to test batch retrieval
    for _ in range(5):
        val, policy = net(history)
        val_scalar = val.item()
        action = torch.randint(0, nactions, (nactions,))
        buffer.push(board, val_scalar, action)
    
    # Test get_batch method
    batch_size = 3
    states, values, actions = buffer.get_batch(batch_size)
    
    # Verify batch shapes and types
    assert len(states) == batch_size
    assert values.shape == (batch_size, 1)
    assert actions.shape == (batch_size, nactions)
    assert isinstance(values, torch.Tensor)
    assert isinstance(actions, torch.Tensor)

# 1. test the mcts returns a valid policy with expected shape 
def test_mcts_sanity_check(env: ChessEnv, net: ChessNet, nactions: int): 
    cfg = MCTSConfig(
        num_sims=10, 
        cpuct=1.0, 
        noise_scale=1.0, 
    )

    empirical_policy = MCTS(cfg, env.board.copy(), net, env.clone(), nactions)

    assert empirical_policy.shape == (nactions,)
    assert torch.all(empirical_policy >= 0)
    assert torch.isclose(empirical_policy.sum(), torch.tensor(1.0), atol=1e-5)

    legal_indices = [move2index(env.board, m) for m in env.board.legal_moves]
    # check legality 
    for i in range(nactions):
        if i in legal_indices: 
            pass 
        else: 
            assert empirical_policy[i] == 0


# 2. test the tree is growing correctly during search 
def test_mcts_tree_growth(env: ChessEnv, net: ChessNet, nactions: int):
    cfg = MCTSConfig(
        num_sims=5,
        cpuct=1.0,
        noise_scale=0.1,
    )
    
    # Get the root node before running MCTS
    root = get_root(env.board.copy(), net, nactions, noise_scale=cfg.noise_scale)
    
    # Initially, root should have no children
    assert len(root.children) == 0
    assert root.count == 0
    
    # Run MCTS with a small number of simulations
    empirical_policy = MCTS(cfg, env.board.copy(), net, env.clone(), nactions)
    
    # After MCTS, we need to get the root again to check its state
    # Since MCTS creates its own root internally, we'll test indirectly
    # by checking that the empirical policy has non-zero values for legal moves
    legal_indices = [move2index(env.board, m) for m in env.board.legal_moves]
    
    # At least some legal moves should have been explored (non-zero probability)
    explored_moves = sum(1 for i in legal_indices if empirical_policy[i] > 0)
    assert explored_moves > 0, "MCTS should explore at least one legal move"
    
    # Test with more simulations to ensure tree grows
    cfg_more_sims = MCTSConfig(
        num_sims=20,
        cpuct=1.0,
        noise_scale=0.1,
    )
    
    empirical_policy_more = MCTS(cfg_more_sims, env.board.copy(), net, env.clone(), nactions)
    
    # With more simulations, we should explore more moves or have more refined probabilities
    explored_moves_more = sum(1 for i in legal_indices if empirical_policy_more[i] > 0)
    
    # Either more moves explored or similar number but different distribution
    assert explored_moves_more >= explored_moves, "More simulations should not reduce exploration"