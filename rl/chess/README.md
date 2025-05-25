## Setup

This is currently a work in progress! 

You'll need to 

`pip install python-chess` 

in addition to the packages in the main `README.md` since we use that to do 
chess-specific board manipulations/computations, etc.

## TODOs

- [x] Understand how AlphaZero represents board
    - [x] Write `board2input` in `utils.py` to take `chess.Board` to input tensors for the `ChessNet` in `model.py` 
    - [x] Implement masking on policy forward pass in `ChessNet` in `model.py` 
- [ ] Implement environment based on `chess.Board` that supports `.reset`, `.step`, and `.clone`. 
- [ ] Implement buffer for value/policy nets that MCTS can feed. 
- [ ] Write `train.py` skeleton. 
- [ ] Implement MCTS. 
- [ ] Implement dynamic batching to make sure forward/backward on networks is constantly high-batch. 

## Goal

- `model.py` defines the convolutional ResNet that we use as our architecture, with a policy and value head. 
- `MCTS.py` defines the search logic taking a state and outputting an optimal action. 
- `config.py` defines key dataclasses and configuration objects that make the rest of the code clean. 
- `buffer.py` implements the buffer that stores experiences collected during MCTS and used to train both the policy/value networks. 
- `utils.py` defines important helper functions we use throughout. 
- `self_play.py` defines the self-play logic. 
- `eval.py` takes in a path for trained weights, outputs Elo by playing many games against a model whose Elo is known.
- `train.py` is the file to run, outlining the training loop and assimilating all other files.  

## Lessons Learned 

TODO 