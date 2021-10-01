import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Reshape, Activation, BatchNormalization, Conv2D, Flatten, ConvLSTM2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy
import copy
import random as rand

# Early stopping callback to monitor loss
callback = EarlyStopping(monitor='loss', min_delta=0.0001, patience=20)

SEED = 1000
tf.random.set_seed(SEED)
numpy.random.seed(SEED)
rand.seed(SEED)

from Board import Board

class Net:

    def __init__(self, n, modelName, training_boards=[], givenModel=None):
        """
        Parameters:
          n - the size of our game boards
          training_boards - a list of (board, value) pairs
            to train our model on
        """
        self.parameters = {
            "layer1 activ func" : 'relu',
            "layer2 activ func" : 'relu',
            "output activ func" : 'linear',
            "loss func" : 'mean_squared_error',
            "num epochs" : 1000000000, # Early stopping will end *way* sooner than this
            "shuffle" : True,
            "batch size" : 32,
            "loss" : None,
            "accuracy" : None,

            # For the convolutional net
            "dropout" : 0.3,
            "num_channels" : 512
        }
        # Activation functions
        self.layer1actfn = self.parameters["layer1 activ func"]
        self.layer2actfn = self.parameters["layer2 activ func"]
        self.layer_out_actfn = self.parameters["output activ func"]
        
        # Initial untrained loss function
        self.loss_fn = self.parameters["loss func"] #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        #########################################
        # First, we check if a trained model is
        # already given.
        self.modelName = modelName
        if givenModel != None:
            self.model = givenModel
            self.modelName = modelName
            return # skip training phase; model is already trained
        elif modelName == 'feedforward':
            self.make_feedforward_net(n, training_boards)
        elif modelName == 'convlstm':
            self.make_convlstm(n, training_boards)

    def make_feedforward_net(self, n, training_sequences=[]):
        """
        Our control group:  A feedforward network.
        
        We just construct a regular feedforward ANN with a single hidden layer.
        We train it on the *last* board from each sequence (since it can't read
        sequential data), but we use the same evaluation that we obtained from
        sequential analysis of the boards.
        """
        # First build the net
        input_layer = Input(shape=(4 + (2*n+2) + 3*(n**2),))
        first_layer = Dense(int(n**2), activation='relu')(BatchNormalization(axis=1)(input_layer))
        output_layer = Dense(1, activation=self.layer_out_actfn)(first_layer)
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        # Then train it on the last board in each sequence
        if training_sequences != []:
            x_train = []
            y_train = []
            for sequence, value in training_sequences:
                x_train.append(numpy.asarray(self.parse_board_feedforward(sequence.current_board)))
                y_train.append(numpy.asarray([value]))
            self.train(numpy.asarray(x_train), numpy.asarray(y_train))

    def make_convlstm(self, n, training_sequences=[]):
        """
        A convolutional LSTM network
        Following this tutorial:
        https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
        """
        input_layer = Input(shape=(None, n, n))
        x_image = Reshape((None, n, n, 1))(input_layer)
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(ConvLSTM2D(self.parameters["num_channels"], 3, padding='same')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(ConvLSTM2D(self.parameters["num_channels"], 3, padding='same')(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(ConvLSTM2D(self.parameters["num_channels"], 3, padding='same')(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(ConvLSTM2D(self.parameters["num_channels"], 3, padding='valid')(h_conv3)))
        h_conv4_flat = Flatten()(h_conv4)

        s_fc1 = Dropout(self.parameters["dropout"])(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))
        s_fc2 = Dropout(self.parameters["dropout"])(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))

        linear_output = Dense(1, activation=self.layer_out_actfn)(s_fc2)
        self.model = tf.keras.Model(inputs=input_layer, outputs=[linear_output])
        self.model.compile(loss=['mean_squared_error'], optimizer='adam')

        # if training_boards != []:
        #     x_train = []
        #     y_train = []
        #     for board, value in training_boards:
        #         x_train.append(numpy.asarray(self.parse_sequence(n, board)))
        #         y_train.append(numpy.asarray([value]))

        #     self.train(numpy.asarray(x_train), numpy.asarray(y_train))

    def parse_board_feedforward(self, board):
        """
        Takes in a board and returns a list of features representing that
        board in a format suitable as input to our neural network.
        """
        parsed = []
        taken = board.get_taken()
        dead = board.get_dead()
        available = board.get_available_not_dead()

        # We get the tempo bit from the board, as well
        # as the total number of taken, dead, and available squares.
        parsed += [board.turn, len(taken), len(dead), len(available)]

        parsed += [int(c % 2 == 0) for c in board.row_count]
        parsed += [int(c % 2 == 0) for c in board.col_count]
        parsed += [int(c % 2 == 0) for c in board.diag_count]

        # We then construct a map of taken squares on the board.
        for i in range(board.size):
            for j in range(board.size):
                if (i, j) in taken:
                    parsed.append(1)
                else:
                    parsed.append(0)
        
        # We do the same for dead squares on the board
        for i in range(board.size):
            for j in range(board.size):
                if (i, j) in dead:
                    parsed.append(1)
                else:
                    parsed.append(0)

        # We do the same for available squares on the board
        for i in range(board.size):
            for j in range(board.size):
                if (i, j) in available:
                    parsed.append(1)
                else:
                    parsed.append(0)

        return parsed

    def parse_sequence(self, n, board_sequences):
        """
        Takes in a sequence of boards and returns a list of board representations
        in a format suitable as input to our LSTM.
        """
        parsed = []

        for seq in board_sequences:
            parsed_seq = []

            for board in seq.boardlist:
                parsed_board = []
                taken = board.get_taken()

                # We then construct a map of taken squares on the board.
                for i in range(board.size):
                    sub = []
                    for j in range(board.size):
                        if (i, j) in taken:
                            sub.append(1)
                        else:
                            sub.append(0)

                    parsed_board.append(sub)
                parsed_seq.append(board)
            parsed.append(seq)

        return parsed

    def train(self, x_train, y_train):
        self.model.compile(optimizer='adam',
            loss=self.loss_fn,
            metrics=['mean_squared_error']) # Note: accuracy is only a good metric for classification problems.
        self.model.fit(x_train, y_train, batch_size=self.parameters["batch size"], 
            epochs=self.parameters["num epochs"],
            shuffle=self.parameters["shuffle"],
            callbacks=[callback])

    def evaluate_board_sequence(self, board_sequence): 
        # Feedforward prediction
        if self.modelName == "feedforward":
            x_test = self.parse_board_feedforward(board_sequence.current_board)
            prediction = self.model.predict(numpy.asarray([numpy.asarray(x_test)]))

        # Convolutional prediction
        # if self.modelName == 'convolutional':
        #     x_test = self.parseBoard_Convolutional(board)
        #     linear_output = self.model.predict(numpy.asarray([numpy.asarray(x_test)])) # pi, v
        #     prediction = linear_output

        return prediction[0][0]

    def next_move(self, board_sequence, nondeterminism=False):
        opn = board_sequence.current_board.get_available()
        dead = board_sequence.current_board.get_dead()
        choices = [c for c in opn if c not in dead]
        
        # If there are no non-dead squares, the net just picks
        # one at random to lose by.
        if choices == []:
            return rand.choice(opn)
        
        # Otherwise, we play all available squares, and pick
        # the one with the *lowest* score (since we're playing
        # hypothetical boards for the *opponent*)
        scores = dict()
        for pos in choices:
            cpy = copy.deepcopy(board_sequence)
            next_board = copy.deepcopy(cpy.current_board)
            next_board.update(pos)
            cpy.append(next_board)

            value = self.evaluate_board_sequence(cpy)
            scores[pos] = value
        m1 = min(scores, key=lambda key: scores[key])

        if nondeterminism and len(scores) > 1:
            # Get the next lowest score, and randomly select
            # between the two (with heavy bias towards the 
            # winning move).  This introduces an element of
            # nondeterminism in game-play.
            scores_copy = dict(scores)
            del scores_copy[m1]
            m2 = min(scores_copy, key=lambda key: scores[key])
            return rand.choices([m1, m2], weights=[0.95, 0.05], k=1)[0]
        else:
            # There's only one move to make, so make it.
            return m1


if __name__ == "__main__":
    # TODO: run a couple of tests just to make sure the neural
    #   network is designed correctly + takes in input correctly

    # First, we set up some test boards, and make up
    # arbitrary values for them.  In practice, these
    # values will be calculated using MCTS.
    N = 4
    board1 = Board(N)
    board1.update((0, 0))
    board1.update((0, 2))
    board1.update((1, 1))
    board1.update((1, 3))
    board1.update((0, 2))
    board1.update((2, 0))
    board1.update((2, 2))
    board1.update((3, 1))
    board1.turn = 0

    board2 = Board(N)
    board2.update((0, 0))
    board2.update((1, 0))
    board2.update((2, 0))
    board2.update((2, 1))
    board2.update((1, 2))
    board2.turn = 1
    
    board3 = Board(N)
    board3.update((0, 0))
    board3.update((0, 1))
    board3.update((0, 3))
    board3.update((1, 1))
    board3.update((2, 1))
    board3.update((2, 3))
    board3.update((3, 3))
    board3.turn = 0

    val1 = 1.0
    val2 = 0.0
    val3 = 0.0

    # We feed these boards as training data into our neural network.
    net = FeatureLayeredNet(N, [
        (board1, val1) 
        # (board2, val2),
        # (board3, val3)
    ])
