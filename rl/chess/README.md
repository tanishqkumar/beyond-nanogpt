## Setup

This is currently a work in progress! 

You'll need to 

`pip install python-chess pytest` 

in addition to the packages in the main `README.md`.

## TODOs

- [x] Understand how AlphaZero represents board
    - [x] Write `board2input` in `utils.py` to take `chess.Board` to input tensors for the `ChessNet` in `model.py` 
    - [x] Implement masking on policy forward pass in `ChessNet` in `model.py` 
- [x] Implement environment based on `chess.Board` that supports `.reset`, `.step`, and `.clone`. 
- [x] Implement buffer for value/policy nets that MCTS can feed. 
- [ ] Write `train.py` skeleton. 
- [ ] Implement MCTS. 
- [ ] Flesh out `train.py` into a full (if slow) training loop. 
- [ ] Optimizate for speed 
    - [ ] Profile to find slowest parts. 
    - [ ] Go through all above, seeing what can be easily vectorized. 
    - [ ] Implement dynamic batching for rollouts. 

## Goal

- `model.py` defines the convolutional ResNet that we use as our architecture, with a policy and value head. 
- `MCTS.py` defines the search logic taking a state and outputting an optimal action. 
- `config.py` defines key dataclasses and configuration objects that make the rest of the code clean. 
- `buffer.py` implements the buffer that stores experiences collected during MCTS and used to train both the policy/value networks. 
- `utils.py` defines important helper functions we use throughout. 
- `env.py` sets up a wrapper on `chess.Board` allowing us to treat it like an RL environment. 
- `eval.py` takes in a path for trained weights, outputs Elo by playing many games against a model whose Elo is known.
- `test_all.py` runs unit and integration tests, use `pytest test_all.py -v` to launch.  
- `train.py` is the file to run, outlining the training loop and assimilating all other files.  

<!-- ## Lessons Learned 

TODO  -->