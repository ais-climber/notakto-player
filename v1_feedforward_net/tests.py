# Unit testing for this quals question

from Board import *
from FeatureLayeredNet import *
from strategies import *
import unittest

if __name__ == "__main__":
    
    # Testing MCTS:
    print("play_until_end: Board is completed.")
    b = Board(4)
    b.play_until_end(greedy)
    assert(b.game_over())
