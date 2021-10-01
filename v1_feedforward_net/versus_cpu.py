from Board import Board
from strategies import *
import random as rand
import time

# Play game against a slightly competent opponent
def player_vs_opponent(next_move=greedy):
    board = Board(7)
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


# Preliminary tests
# For comparing our rule-based and greedy strategies
def preliminary_tests(strategy_1, strategy_2, num_trials):
    print(f"{num_trials} trials")
    for size in [3]:#range(3, 7):
        wins1 = 0
        wins2 = 0

        for trial in range(num_trials):
            #print("Trial ", trial, "...")
            board = Board(size)

            # Strategy 1 always goes first, by convention.
            turn = 0
            
            while not board.game_over():
                # Strategy #1
                if turn == 0:
                    p = strategy_1(board)
                    board.update(p)
                    turn = 1
                
                # Strategy #2
                elif turn == 1:
                    p = strategy_2(board)
                    board.update(p)
                    turn = 0

            # Update winner count
            if turn == 0:
                wins1 += 1
            elif turn == 1:
                wins2 += 1
        
        print(f"Board Size {size}")
        print(f"1 - {strategy_1.__name__}: {100*round(wins1/num_trials, 3)}%")
        print(f"2 - {strategy_2.__name__}: {100*round(wins2/num_trials, 3)}%")
        print()


if __name__ == "__main__":
    # Run if you want to play the game against a basic cpu opponent
    player_vs_opponent()