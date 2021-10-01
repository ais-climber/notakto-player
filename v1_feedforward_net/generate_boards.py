from Board import Board
from strategies import *
import random as rand
import copy

rand.seed(1000)

# Sample sizes for an nxn board
# n  #unwinned boards   #total boards
# 1  1                  2
# 2  5                  16
# 3  460                512
# 4  65536
# 5  33554432
# 6  68719476736
# 7  562949953421312
# 8  18446744073709551616
# 9  2417851639229258349412352
# 10 1267650600228229401496703205376
# 11 2658455991569831745807614120560689152
# 12 22300745198530623141535718272648361505980416

INITIAL_NUM_BOARDS = {
    2 : 1, # Produces all 5 possible boards
    3 : 12, # Produces 155 boards, i.e. ~200
    4 : 60, # Produces 2378 boards, i.e. ~2000 
    5 : 125, # Prodcues 9958 boards, i.e. ~10000
    6 : 625, # Produces _ boards, i.e. ~100000
    7 : 3125,
    8 : 6656,
    9 : 13719,
    10 : 28220,
    11 : 57779,
    12: 117696,
    13 : 238619
}

def remove_repeats(board_list):
    no_repeats = []
    for b in board_list:
        if b not in no_repeats:
            no_repeats.append(b)

    return no_repeats

def gen_initial_boards(N):
    """
    Note that the way we do this does in fact handle early-game
    and late-game configurations.
    """
    # We initialize our list of boards with an empty board.
    board_list = [Board(N)]
    num_games = INITIAL_NUM_BOARDS[N] #int(INITIAL_NUM_BOARDS[N] / (N**1.80))

    for game in range(num_games):
        board = Board(N)

        # In our initial set of games, we always start
        # with the player going first.
        board.turn = 0

        while not board.game_over():
            # Player #1
            if board.turn == 0:
                p = greedy(board)
                board.update(p)
                board.turn = 1
            
            # Player #2
            elif board.turn == 1:
                p = greedy(board)
                board.update(p)
                board.turn = 0
            
            # Ensure no repeats and ensure game is not already over
            c = copy.deepcopy(board)
            if c not in board_list and not c.game_over():
                board_list.append(c)

    # We then remove any repeats from the initial set of boards.
    return board_list

def neighborhood(board_list):
    """
    Get boards in the neighborhood of the boards in board_list.

    What this means, precisely, is that, if possible, we add
    up to two randomly placed X's and remove up to two randomly
    placed X's from each board.

    This may introduce repeated boards, which we go ahead and remove.
    """
    neighborhood = copy.deepcopy(board_list)

    for b in board_list:
        # We attempt to add an X:
        opn = b.get_available()
        dead = b.get_dead()

        # Pick a random non-dead square, if there is one.
        choices = [c for c in opn if c not in dead]
        if choices != []:
            bX = copy.deepcopy(b)
            bX.update(rand.choice(choices))

            if bX not in neighborhood:
                neighborhood.append(bX)

            # We attempt to add a second X
            opn = bX.get_available()
            dead = bX.get_dead()
            choices = [c for c in opn if c not in dead]
            if choices != []:
                bXX = copy.deepcopy(bX)
                bXX.update(rand.choice(choices))

                if bXX not in neighborhood:
                    neighborhood.append(bXX)
        
        # We now attempt to remove an X:
        tkn = b.get_taken()
        choices = [c for c in tkn]
        if choices != []:
            bX = copy.deepcopy(b)
            c = rand.choice(choices)
            bX.game_board[c[0]][c[1]] = "_"

            if bX not in neighborhood:
                neighborhood.append(bX)

            # We attempt to remove a second X
            tkn = bX.get_taken()
            choices = [c for c in tkn]
            if choices != []:
                bXX = copy.deepcopy(b)
                c = rand.choice(choices)
                bXX.game_board[c[0]][c[1]] = "_"

                # TODO: fix error here
                if bXX not in neighborhood:
                    neighborhood.append(bXX)

    return neighborhood


def flip_turns(board_list):
    a = copy.deepcopy(board_list)

    for b1 in a:
        b1.update_turn()

        if b1 not in a:
            a.append(b1)
    
    return a

if __name__ == "__main__":
    init = gen_initial_boards(6)
    print("Done initial boards")
    nbhd = neighborhood(init)
    print("Done neighborhood")
    flipped = flip_turns(nbhd)
    print("Finished flipping!")
    # print("Flipped game boards:")
    # for board in flipped:
    #     print(board)

    # Test for no repeats
    # flag = False
    # for i in range(len(flipped)):
    #     for j in range(len(flipped)):
    #         if i != j and flipped[i] == flipped[j]:
    #             flag = True
    # print(f"Are there duplicates?  {flag}")

    print(len(flipped))
