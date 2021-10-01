from Board import Board
import random as rand
rand.seed(1)

# Strategy for randomly generating boards:
def greedy(game_board):
    """
    Returns a randomly selected board coordinate,
    so long as that choice would not end the game.
    """
    opn = game_board.get_available()
    dead = game_board.get_dead()

    # Pick a random non-dead square, if there is one.
    choices = [c for c in opn if c not in dead]
    if choices != []:
        return rand.choice(choices)
    else:
        return rand.choice(opn)

def random_selection(game_board):
    """
    Returns a randomly selected board coordinate from
    those available.
    """
    return rand.choice(game_board.get_available())
