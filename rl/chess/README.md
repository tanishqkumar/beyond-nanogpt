## Setup

You'll need to 

`pip install python-chess` 

in addition to the packages in the main `README.md` since we use that to do 
chess-specific board manipulations/computations, etc.

## Overview

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