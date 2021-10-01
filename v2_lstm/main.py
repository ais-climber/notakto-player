from Board import Board
from BoardSequence import BoardSequence
from Nets import Net
from strategies import *
from generate_games import *

import tensorflow as tf
import pickle
import sys

def player_vs_opponent(N, next_move):
    board = Board(N)
    turn = 0
    print(board)
    
    while not board.game_over():
        # Player's turn
        if turn == 0:
            print("Player's move")

            inp = input("> ")
            inp = inp.replace(" ", "").replace(",", "")
            t = tuple(inp)
            p = (int(t[0]), int(t[1]))

            if board.is_already_taken(p):
                continue

            board.update(p)
            turn = 1
            print(board)
        
        # Opponent's turn
        elif turn == 1:
            time.sleep(0.25)
            print("Opponent's move")

            p = next_move(board)
            board.update(p)
            turn = 0
            print(board)

    # Determine the winner
    if turn == 0:
        winner = "Player"
    elif turn == 1:
        winner = "Opponent"
    print(f"Game over!  {winner} wins!")


def pit(N, num_trials, strategy_1, strategy_2):
    print(f"{num_trials} trials on a {N}x{N} board:")
    wins1 = 0
    wins2 = 0
    game_sequence = BoardSequence(Board(N))

    for trial in range(num_trials):
        board = Board(N)

        # Strategy 1 always goes first, by convention.
        turn = 0
        while not board.game_over():
            # Strategy #1
            if turn == 0:
                p = strategy_1(game_sequence)
                board.update(p)
                turn = 1
                game_sequence.append(copy.deepcopy(board))
            
            # Strategy #2
            elif turn == 1:
                p = strategy_2(game_sequence)
                board.update(p)
                turn = 0
                game_sequence.append(copy.deepcopy(board))

        # Update winner count
        if turn == 0:
            wins1 += 1
        elif turn == 1:
            wins2 += 1

    print(f"1 - {strategy_1.__name__}: {100*round(wins1/num_trials, 3)}%")
    print(f"2 - {strategy_2.__name__}: {100*round(wins2/num_trials, 3)}%")
    print()


if __name__ == "__main__":
    options = ["player_versus_greedy"]

    # Preset game modes
    if len(sys.argv) == 3 and sys.argv[1] in options and sys.argv[2].isdigit():
        game_mode = sys.argv[1]
        N = sys.argv[2]

        # Player versus greedy opponent
        if game_mode == "player_versus_greedy":
            player_vs_opponent(N, next_move)

    # Default tests
    elif len(sys.argv) == 1:
        N = 3
        training_set = generate_boards(N)
        print(len(training_set))
        
        #feedforward = Net(N, 'feedforward', training_set)
        convlstm = Net(N, 'convlstm', training_set)

        print("Testing...")

        print("feedforward v itself")
        pit(N, 100, feedforward.next_move, feedforward.next_move)

        print("feedforward v greedy")
        pit(N, 100, feedforward.next_move, greedy)
        pit(N, 100, greedy, feedforward.next_move)

        print("Finished testing!...")
        
        filename = "_model_" + str(N)
        # convnet.model.save("convnet"+filename)
        # featnet.model.save("featnet"+filename)

        import os
        # Play a 'done' sound for long waits
        # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.09, 197.58))
        # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.09, 263.74))
        # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.09, 352.04))
        # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.15, 395.16))

    else:
        print("Usage:   $ python3 main.py <game_mode> <N>")
