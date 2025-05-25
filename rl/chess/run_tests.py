from model import ChessNet 
import chess 
import torch, torch.nn as nn, torch.nn.functional as F
from utils import move2index

# this implies the diff is bounded 

def test_model_size():
    print(f'Initializing ChessNet...')
    mtl = 119  # in channels, M*T + L, see paper on how they represent a cheess board 
               # to give it as input into their alphazero conv neural net 
    net = ChessNet(in_ch=mtl)
    num = sum(p.numel() for p in net.parameters())
    print(f'Net has {num/1e6:.1f}M parameters!')
    return net, mtl

def test_forward_pass(net, nactions):
    board = chess.Board()
    history = [board]
    
    # Test with board history
    v, l = net(history)
    assert v.shape == (1, 1) and l.shape == (1, nactions)
    
    # Test with single board
    v, l = net(board)
    assert v.shape == (1, 1) and l.shape == (1, nactions)
    print(f'Forward test passed with real input!')

def test_move_masking(net):
    board = chess.Board()
    legal_move = chess.Move.from_uci("e2e4")
    illegal_move = chess.Move.from_uci("e2e6")
    
    values, logits = net(board)
    
    legal_idx = move2index(board, legal_move)
    illegal_idx = move2index(board, illegal_move)
    
    # Verify legality is correctly masked 
    assert torch.isinf(logits[0, illegal_idx]) and logits[0, illegal_idx] < 0, "Illegal move should be masked with -inf"
    assert not torch.isinf(logits[0, legal_idx]), "Legal move should not be masked"
    
    print(f'Masking test passed!')


if __name__ == "__main__":
    nactions = 4672
    net, mtl = test_model_size()
    test_forward_pass(net, nactions)
    test_move_masking(net)
    
    print(f'--' * 20)
    print(f'ALL TESTS PASSED.')