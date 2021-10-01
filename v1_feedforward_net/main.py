from Board import Board
from FeatureLayeredNet import FeatureLayeredNet
from strategies import *
from generate_boards import *
import tensorflow as tf
from versus_cpu import *
import pickle

if __name__ == "__main__":
    N = 3
    
    # flipped = []
    # with open('size_4_boards.pkl', 'rb') as inp:
    #     flipped = pickle.load(inp)

    #Board generation code
    init = gen_initial_boards(N)
    nbhd = neighborhood(init)
    flipped = flip_turns(nbhd)

    print(len(flipped))
    #input()

    # Phase 1: Training on randomly played boards
    #print("PHASE 1")
    print("Generating training boards...")
    training_set = []
    for b in flipped:
        value = b.MCTS(random_selection)
        training_set.append((b, value))

    # print("Training the featured net...")
    # featnet = FeatureLayeredNet(N, training_set, modelName='featured_convolutional')
    # print("Training the convolutional net...")
    # convnet = FeatureLayeredNet(N, training_set, modelName='convolutional')

    # cmodel = tf.keras.models.load_model("Models/convnet_model_"+str(N), compile=True)
    # convnet = FeatureLayeredNet(N, training_boards=[], givenModel=cmodel, modelName='convolutional')
    trialnet = FeatureLayeredNet(N, training_set)
    
    # Optional play against a human
    #player_vs_opponent(net.next_move)

    print("Testing...")

    ######################################
    print("TRIAL v Itself:")
    preliminary_tests(
        (lambda board: trialnet.next_move(board, nondeterminism=True)), 
        (lambda board: trialnet.next_move(board, nondeterminism=True)), 100)

    print("TRIAL v Greedy:")
    preliminary_tests(trialnet.next_move, greedy, 100)
    preliminary_tests(greedy, trialnet.next_move, 100)
    
    print("Finished testing!...")
    ######################################

    # print("CONV v Itself:")
    # preliminary_tests(convnet.next_move, convnet.next_move, 100)

    # print("CONV v Random:")
    # preliminary_tests(convnet.next_move, random_selection, 100)
    # preliminary_tests(random_selection, convnet.next_move, 100)

    # print("CONV v Greedy:")
    # preliminary_tests(convnet.next_move, greedy, 100)
    # preliminary_tests(greedy, convnet.next_move, 100)

    # print("FEAT v Itself:")
    # preliminary_tests(featnet.next_move, featnet.next_move, 100)

    # print("FEAT v Random:")
    # preliminary_tests(featnet.next_move, random_selection, 100)
    # preliminary_tests(random_selection, featnet.next_move, 100)

    # print("FEAT v Greedy:")
    # preliminary_tests(featnet.next_move, greedy, 100)
    # preliminary_tests(greedy, featnet.next_move, 100)

    # print("CONV v FEAT:")
    # preliminary_tests(convnet.next_move, featnet.next_move, 100)
    # preliminary_tests(featnet.next_move, convnet.next_move, 100)
    # print("Finished testing!...")

    filename = "_model_" + str(N)
    # convnet.model.save("convnet"+filename)
    # featnet.model.save("featnet"+filename)

    import os
    # Play a 'done' sound for long waits
    # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.09, 197.58))
    # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.09, 263.74))
    # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.09, 352.04))
    # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.15, 395.16))
