from Board import Board
from BoardSequence import BoardSequence
from strategies import *
import random as rand
import copy

rand.seed(1000)

NUM_GAMES = {
    2 : 5, # Produces all 5 possible boards
    3 : 3,#1000, #753
    4 : 500, #3224
    5 : 500, #
    6 : 500, #
    7 : 500,
    8 : 6656,
    9 : 13719,
    10 : 28220,
    11 : 57779,
    12: 117696,
    13 : 238619
}

def play_and_evaluate(N, next_move):
    """
    A function to generate some sequences of boards via play
    and to evaluate these sequences.

    Note that the way we do this does in fact handle early-game
    and late-game configurations.
    """
    games = []
    evaluations = []

    for game in range(NUM_GAMES[N]):
        # Initialize the sequence of games with the player going first.
        this_sequence = BoardSequence(Board(N))
        this_sequence.current_board.turn = 0

        while not this_sequence.current_board.game_over():
            seq = copy.deepcopy(this_sequence)
            board = copy.deepcopy(this_sequence.current_board)

            # Player #1
            if board.turn == 0:
                p = next_move(seq)
                board.update(p)
                board.turn = 1
                seq.append(board)
                this_sequence.append(board)

                if seq not in games and not seq.current_board.game_over():
                    value = seq.evaluate(next_move)
                    games.append(seq)
                    evaluations.append((seq, value))
            
            # Player #2
            elif board.turn == 1:
                p = next_move(seq)
                board.update(p)
                board.turn = 0
                seq.append(board)
                this_sequence.append(board)

                if seq not in games and not seq.current_board.game_over():
                    value = seq.evaluate(next_move)
                    games.append(seq)
                    evaluations.append((seq, value))           

        # # Truncate the play that ended the game    
        # this_sequence = this_sequence.truncate(len(this_sequence.boardlist)-2)

        # # Ensure no repeats and ensure game is not already over.
        # # Then, evaluate the board sequence.
        # if this_sequence not in games and not this_sequence.current_board.game_over():
        #     value = this_sequence.evaluate(next_move)
        #     games.append((this_sequence, value))

    return evaluations


def flip_turns(board_sequences, next_move):
    # TODO: fix for board *sequences*
    cpy = copy.deepcopy(board_sequences)
    sequences = [c[0] for c in cpy]

    for game in cpy:
        seq = game[0]

        for board in seq.boardlist:
            board.update_turn()

        value = seq.evaluate(next_move)
        if seq not in sequences:
            cpy.append((seq, value))
            sequences.append(seq)
    
    return cpy

def generate_boards(N):
    return play_and_evaluate(N, greedy)

if __name__ == "__main__":
    N = 3
    games = generate_boards(N)

    for game in games:
        for board in game[0].boardlist:
            print(board)
        print(f"SCORE: {game[1]}")

    print(len(games))
    
