"""
A script for producing a heatmap of a Notakto strategy.
For a given strategy S, we play S in a variety of games.
From these games, we collect the most likely responses
of S to a given move m, and make a heatmap of these responses.
"""

from Board import Board
from FeatureLayeredNet import FeatureLayeredNet
from strategies import *
from generate_boards import *

import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def play_against(N, player_1, player_2):
    """
    Plays a single game between player_1 and player_2, on an NxN board.

    Returns two lists, one for player_1 and the other for player_2.
    Each list consists of
        (their, my)
    pairs, where 'their' is the move the other player had made, and
    'my' is the move this player made in response.

    We also return alongside these the winner, so that we can e.g.
    filter games that were won by player_1.
    """
    board = Board(N)
    p1responses = []
    p2responses = []

    # Player 1 always goes first, by convention.
    turn = 0
    previous_move = None
    
    while not board.game_over():
        # Player #1
        if turn == 0:
            this_move = player_1(board)
            board.update(this_move)
            turn = 1

            p1responses.append((previous_move, this_move))
            previous_move = this_move
        
        # Player #2
        elif turn == 1:
            this_move = player_2(board)
            board.update(this_move)
            turn = 0

            p2responses.append((previous_move, this_move))
            previous_move = this_move

    if turn == 0:
        winner = "player_1"
    elif turn == 1:
        winner = "player_2"

    return (winner, p1responses, p2responses)

if __name__ == "__main__":
    N = 3
    num_games = 1000
    winning_choice = "player_1"
    init = gen_initial_boards(N)
    nbhd = neighborhood(init)
    flipped = flip_turns(nbhd)

    # Initialize net players
    fmodel = tf.keras.models.load_model("Models/featnet_model_"+str(N), compile=True)
    cmodel = tf.keras.models.load_model("Models/convnet_model_"+str(N), compile=True)
    featnet = FeatureLayeredNet(N, training_boards=[], givenModel=fmodel, modelName='featured_convolutional')
    convnet = FeatureLayeredNet(N, training_boards=[], givenModel=cmodel, modelName='convolutional')

    # Play some games and mark down the feature net's responses
    player_responses = []
    for i in range(num_games):
        print(i)
        # Only collect games where greedy won (this reveals its winning strategy)

        # Against greedy
        winner, p1res, p2res = play_against(N, featnet.next_move, greedy)
        #winner, p1res, p2res = play_against(N, greedy, featnet.next_move)
        if winner == winning_choice:
            player_responses += p1res

    # Build up a dictionary mapping:
    #    possible opponent squares --> possible response positions, with % cases where we respond in this way.
    mapping = dict()
    for (their, my) in player_responses:
        # Initialize if we haven't seen these moves before
        if their not in mapping.keys():
            mapping[their] = {(x, y) : 0 for x in range(N) for y in range(N)}
        # if my not in mapping[their].keys():
        #     mapping[their][my] = 0
        
        # Increment the number of times we gave this particular response.
        mapping[their][my] += 1

    # Pretty display the responses to each move as a heat map
    print("Displaying and saving figures...")
    for move in mapping.keys():
        ser = pd.Series(list(mapping[move].values()),
                    index=pd.MultiIndex.from_tuples(mapping[move].keys()))
        df = ser.unstack().fillna(0)
        ax = sns.heatmap(df, annot=True, cmap="Blues", cbar=False)

        # Draw a red square around the current move
        # (we have to flip the coordinates, since the heatmap goes down then right,
        #  whereas plotting goes right then down.)
        if move != None:
            rect = plt.Rectangle((move[1], move[0]), 1,1, color="red", linewidth=3, fill=False, clip_on=False)
            ax.add_patch(rect)

        plt.axes().set_title("In Response to " + str(move))
        plt.savefig("Heatmaps/heatmap_" + str(N) + "x" + str(N) + "_response_to_" + str(move) + ".png")
        plt.show()



