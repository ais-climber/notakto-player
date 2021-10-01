from Board import Board
from BoardSequence import BoardSequence
import random as rand
rand.seed(1)

# Strategy for randomly generating boards:
def greedy(board_sequence):
    """
    Returns a randomly selected board coordinate,
    so long as that choice would not end the game.
    """
    game_board = board_sequence.current_board
    opn = game_board.get_available()
    dead = game_board.get_dead()

    # Pick a random non-dead square, if there is one.
    choices = [c for c in opn if c not in dead]
    if choices != []:
        return rand.choice(choices)
    else:
        return rand.choice(opn)

def random_selection(board_sequence):
    """
    Returns a randomly selected board coordinate from
    those available.
    """
    game_board = board_sequence.current_board
    return rand.choice(game_board.get_available())
