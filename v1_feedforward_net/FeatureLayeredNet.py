import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Reshape, Activation, BatchNormalization, Conv2D, Flatten
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

class FeatureLayeredNet:

    def __init__(self, n, training_boards=[], givenModel=None, modelName=None):
        """
        Parameters:
          n - the size of our game boards
          training_boards - a list of (board, value) pairs
            to train our model on
        """
        self.parameters = {
            "layer1 activ func" : 'gelu',
            "layer2 activ func" : 'gelu',
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
        elif modelName == 'convolutional':
            self.modelName = modelName
            self.makeNet_Convolutional(n, training_boards)
        elif modelName == 'featured_convolutional':
            self.modelName = modelName
            self.makeNet_Featured_Convolutional(n, training_boards)
        else:
            #self.makeNet_fullyFeatured(n, training_boards)
            self.makeNet_flat(n, training_boards)
            #self.makeNet_Convolutional(n, training_boards)
            #self.makeNet_Featured_Convolutional(n, training_boards)
            #self.makeNet_Best(n, training_boards)
            #self.makeNet_Better(n, training_boards) # This one seems to just memorize the data and overfit :(

    def makeNet_flat(self, n, training_boards=[]):
        # We seperate the features of the input 
        input_layer = Input(shape=(4 + (2*n+2) + 3*(n**2),))
        first_layer = Dense(int(n**2), activation='relu')(BatchNormalization(axis=1)(input_layer))
        # second_layer = Dense(int(0.5*(n**2)), activation='relu')(BatchNormalization(axis=1)(first_layer))
        # third_layer = Dense(int(n), activation='relu')(BatchNormalization(axis=1)(second_layer))
        output_layer = Dense(1, activation=self.layer_out_actfn)(first_layer)

        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        if training_boards != []:
            x_train = []
            y_train = []
            for board, value in training_boards:
                x_train.append(numpy.asarray(self.parseBoard_Flat(board)))
                y_train.append(numpy.asarray([value]))

            self.train(numpy.asarray(x_train), numpy.asarray(y_train))


    def makeNet_fullyFeatured(self, n, training_boards=[]):
        """
        Function to build the fully featured neural network
        using the tensorflow functional API.
        We train the net as well, since the training call
        relies on the particular architecture chosen.
        """
        # We seperate the features of the input 
        input_A = Input(shape=(4,))
        input_PAR = Input(shape=(2*n+2,))
        input_B = Input(shape=(n**2,))
        input_C = Input(shape=(n**2,))
        input_D = Input(shape=(n**2,))

        tempo_layer = Dense(4, activation=self.layer1actfn)(input_A)
        parity_layer = Dense(2*n+2, activation=self.layer1actfn)(input_PAR)
        taken_squares_layer = Dense(n**2, activation=self.layer1actfn)(input_B)
        dead_squares_layer = Dense(n**2, activation=self.layer1actfn)(input_C)
        available_squares_layer = Dense(n**2, activation=self.layer1actfn)(input_D)

        # We merge the features, and then include layers for high-level feature extraction and output.
        merged_layer1 = Concatenate()([taken_squares_layer, dead_squares_layer, available_squares_layer])
        abstract_layer1 = Dense(2*(n**2), activation=self.layer2actfn)(merged_layer1)
        
        merged_layer2 = Concatenate()([abstract_layer1, parity_layer, tempo_layer])
        abstract_layer2 = Dense(n**2, activation=self.layer2actfn)(merged_layer2)
        
        output_layer = Dense(1, activation=self.layer_out_actfn)(abstract_layer2)

        self.model = tf.keras.Model(inputs=[input_A, input_PAR, input_B, input_C, input_D], outputs=output_layer)

        # Sanity checks
        #self.model.summary()
        #tf.keras.utils.plot_model(self.model, show_shapes=True)

        if training_boards != []:
            input_A = []
            input_PAR = []
            input_B = []
            input_C = []
            input_D = []
            y_train = []
            for board, value in training_boards:
                x_train = self.parseBoardAsInput(board)
                input_A.append(x_train[0])
                input_PAR.append(x_train[1])
                input_B.append(x_train[2])
                input_C.append(x_train[3])
                input_D.append(x_train[4])
                y_train.append(value)

            self.train([
                numpy.asarray(input_A), 
                numpy.asarray(input_PAR),
                numpy.asarray(input_B), 
                numpy.asarray(input_C), 
                numpy.asarray(input_D)
                ], 
                numpy.asarray([numpy.asarray([y]) for y in y_train]))

    def makeNet_Convolutional(self, n, training_boards=[]):
        """
        The AlphaZero TicTacToe net by Evgeny Tyurin, github.com/evg-tyurin
        We're using it here for notakto.
        A convolutional net that reads in the board as an image, and takes
        no other features.
        """
        # Neural Net
        input_layer = Input(shape=(n, n))    # s: batch_size x board_x x board_y

        x_image = Reshape((n, n, 1))(input_layer)                # batch_size  x n x n x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.parameters["num_channels"], 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.parameters["num_channels"], 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.parameters["num_channels"], 3, padding='same')(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.parameters["num_channels"], 3, padding='valid')(h_conv3))) 
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(self.parameters["dropout"])(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(self.parameters["dropout"])(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        
        linear_output = Dense(1, activation=self.layer_out_actfn)(s_fc2)
        pi = Dense(n*n+1, activation='softmax', name='pi')(s_fc2)   # batch_size x "action size"
        v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = tf.keras.Model(inputs=input_layer, outputs=[linear_output])#, pi, v])
        self.model.compile(loss=['mean_squared_error'], optimizer='adam') # 'categorical_crossentropy', 

        if training_boards != []:
            x_train = []
            y_train = []
            for board, value in training_boards:
                x_train.append(numpy.asarray(self.parseBoard_Convolutional(board)))
                y_train.append(numpy.asarray([value]))

            self.train(numpy.asarray(x_train), numpy.asarray(y_train))

    def makeNet_Featured_Convolutional(self, n, training_boards=[]):
        """
        The AlphaZero TicTacToe net by Evgeny Tyurin, github.com/evg-tyurin
        We're using it here for notakto.
        A convolutional net that reads in the board as an image, and takes
        no other features.
        """
        # Neural Net
        input_IMG = Input(shape=(n, n))    # s: batch_size x board_x x board_y
        input_TEMPO = Input(shape=(1,))
        #input_DEAD = Input(shape=(1,))
        input_PAR = Input(shape=(2*n+2,))

        x_image = Reshape((n, n, 1))(input_IMG)                # batch_size  x n x n x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.parameters["num_channels"], 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.parameters["num_channels"], 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.parameters["num_channels"], 3, padding='same')(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.parameters["num_channels"], 3, padding='valid')(h_conv3))) 
        h_conv4_flat = Flatten()(h_conv4)

        
        tempo_layer = Dense(1, activation='relu')(BatchNormalization(axis=1)(input_TEMPO))
        #num_dead_layer = Dense(1, activation=self.layer1actfn)(input_DEAD)
        parity_layer = Dense(2*n+2, activation='relu')(BatchNormalization(axis=1)(input_PAR))
        merged_layer = Concatenate()([tempo_layer, parity_layer, h_conv4_flat])
        abstract_layer = Dense(int(n), activation='relu')(BatchNormalization(axis=1)(merged_layer))

        # s_fc1 = Dropout(self.parameters["dropout"])(Activation('relu')(BatchNormalization(axis=1)(Dense(n)(abstract_layer))))  # batch_size x 1024
        # s_fc2 = Dropout(self.parameters["dropout"])(Activation('relu')(BatchNormalization(axis=1)(Dense(n)(s_fc1))))          # batch_size x 512

        linear_output = Dense(1, activation=self.layer_out_actfn)(abstract_layer)

        self.model = tf.keras.Model(inputs=[input_IMG, input_TEMPO, input_PAR], outputs=linear_output)#, pi, v])
        self.model.compile(loss=['mean_squared_error'], optimizer='adam') # 'categorical_crossentropy', 

        if training_boards != []:
            input_IMG = []
            input_TEMPO = []
            #input_DEAD = []
            input_PAR = []
            y_train = []
            for board, value in training_boards:
                x_train = self.parseBoard_Featured_Convolutional(board)
                input_IMG.append(x_train[0])
                input_TEMPO.append(x_train[1])
                #input_DEAD.append(x_train[2])
                input_PAR.append(x_train[2])
                y_train.append(value)

            self.train([
                numpy.asarray(input_IMG), 
                numpy.asarray(input_TEMPO),
                #numpy.asarray(input_DEAD),
                numpy.asarray(input_PAR)
                ], 
                numpy.asarray([numpy.asarray([y]) for y in y_train]))

    def makeNet_Best(self, n, training_boards=[]):
        input_A = Input(shape=(1,))
        input_B = Input(shape=(1,))
        input_C = Input(shape=(2*n + 2,))
        input_D = Input(shape=(n**2,))

        tempo_layer = Dense(1, activation=self.layer1actfn)(input_A)
        dead_squares_layer = Dense(1, activation=self.layer1actfn)(input_B)
        parity_layer = Dense(int(2*n+2), activation=self.layer1actfn)(input_C)
        positional_layer = Dense(int(n**2), activation=self.layer1actfn)(input_D)

        # We merge the features, and then include layers for high-level feature extraction and output.
        merged_layer = Concatenate()([tempo_layer, dead_squares_layer, parity_layer, positional_layer])
        abstract_layer = Dense(int(n**2), activation=self.layer2actfn)(merged_layer)
        dropout_layer = Dropout(self.parameters["dropout"])(abstract_layer)
        output_layer = Dense(1, activation=self.layer_out_actfn)(dropout_layer)

        self.model = tf.keras.Model(inputs=[input_A, input_B, input_C, input_D], outputs=output_layer)

        if training_boards != []:
            input_A = []
            input_B = []
            input_C = []
            input_D = []
            y_train = []
            for board, value in training_boards:
                x_train = self.parseBoard_Best(board)
                input_A.append(x_train[0])
                input_B.append(x_train[1])
                input_C.append(x_train[2])
                input_D.append(x_train[3])
                y_train.append(value)

            self.train([
                numpy.asarray(input_A), 
                numpy.asarray(input_B), 
                numpy.asarray(input_C), 
                numpy.asarray(input_D)
                ], 
                numpy.asarray([numpy.asarray([y]) for y in y_train]))

    def makeNet_Better(self, n, training_boards=[]):
        input_A = Input(shape=(1,))
        input_B = Input(shape=(1,))
        input_C = Input(shape=(2*n + 2,))
        input_D = Input(shape=(n**2,))
        input_E = Input(shape=(n**2,))
        input_F = Input(shape=(2*n,))

        tempo_layer = Dense(1, activation=self.layer1actfn)(input_A)
        dead_squares_layer = Dense(1, activation=self.layer1actfn)(input_B)
        parity_layer = Dense(int(2*n+2), activation=self.layer1actfn)(input_C)
        rows_layer = Dense(int(n**2), activation=self.layer1actfn)(input_D)
        cols_layer = Dense(int(n**2), activation=self.layer1actfn)(input_E)
        diags_layer = Dense(int(2*n), activation=self.layer1actfn)(input_F)

        # We merge the features, and then include layers for high-level feature extraction and output.
        merged_layer = Concatenate()([tempo_layer, dead_squares_layer, parity_layer, rows_layer, cols_layer, diags_layer])
        abstract_layer = Dense(int(n**2), activation=self.layer2actfn)(merged_layer)
        output_layer = Dense(1, activation=self.layer_out_actfn)(abstract_layer)

        self.model = tf.keras.Model(inputs=[input_A, input_B, input_C, input_D, input_E, input_F], outputs=output_layer)

        if training_boards != []:
            input_A = []
            input_B = []
            input_C = []
            input_D = []
            input_E = []
            input_F = []
            y_train = []
            for board, value in training_boards:
                x_train = self.parseBoard_Better(board)
                input_A.append(x_train[0])
                input_B.append(x_train[1])
                input_C.append(x_train[2])
                input_D.append(x_train[3])
                input_E.append(x_train[4])
                input_F.append(x_train[5])
                y_train.append(value)

            self.train([
                numpy.asarray(input_A), 
                numpy.asarray(input_B), 
                numpy.asarray(input_C), 
                numpy.asarray(input_D),
                numpy.asarray(input_E),
                numpy.asarray(input_F)
                ], 
                numpy.asarray([numpy.asarray([y]) for y in y_train]))

    def parseBoardAsInput(self, board):
        """
        Takes in a board and returns a list representing that
        board in a format suitable as input to our neural network.
        """
        parsed = []
        taken = board.get_taken()
        dead = board.get_dead()
        available = board.get_available_not_dead()

        # We get the tempo bit from the board, as well
        # as the total number of taken, dead, and available squares.
        input_A = [board.turn, len(taken), len(dead), len(available)]

        # I'm going to add to this the parity score for each row, then each
        # column, then finally the diagonal and antidiagonal.
        input_PAR = []
        input_PAR += [int(c % 2 == 0) for c in board.row_count]
        input_PAR += [int(c % 2 == 0) for c in board.col_count]
        input_PAR += [int(c % 2 == 0) for c in board.diag_count]

        # We then construct maps of taken, dead, and available squares
        # on the board.
        input_B = []
        input_C = []
        input_D = []
        for i in range(board.size):
            for j in range(board.size):
                if (i, j) in taken:
                    input_B.append(1)
                else:
                    input_B.append(0)
                
                if (i, j) in dead:
                    input_C.append(1)
                else:
                    input_C.append(0)

                if (i, j) in available:
                    input_D.append(1)
                else:
                    input_D.append(0)

        parsed.append(input_A)
        parsed.append(input_PAR)
        parsed.append(input_B)
        parsed.append(input_C)
        parsed.append(input_D)
        return parsed

    def parseBoard_Flat(self, board):
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

        # I'm going to add to this the parity score for each row, then each
        # column, then finally the diagonal and antidiagonal.
        # NOTE: I tried doing the parity score for *available, but not dead*
        # squares.  I believed this would give a more meaningful measure of the state
        # of the game, but it didn't pan out to decent results.
        # avail_row = [0 for i in range(board.size)]
        # avail_col = [0 for j in range(board.size)]
        # avail_diag = [0, 0]

        # for i in range(board.size):
        #     for j in range(board.size):
        #         if (i, j) in available:
        #             avail_row[i] += 1
        #             avail_col[j] += 1

        #             if board.is_on_diagonal((i, j)):
        #                 avail_diag[0] += 1
        #             if board.is_on_antidiagonal((i, j)):
        #                 avail_diag[1] += 1

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

    def parseBoard_Convolutional(self, board):
        """
        Takes in a board and returns a list of features representing that
        board in a format suitable as input to our neural network.
        """
        parsed = []
        taken = board.get_taken()

        # We then construct a map of taken squares on the board.
        for i in range(board.size):
            sub = []
            for j in range(board.size):
                if (i, j) in taken:
                    sub.append(1)
                else:
                    sub.append(0)
            parsed.append(sub)

        return parsed

    def parseBoard_Featured_Convolutional(self, board):
        """
        Takes in a board and returns a list representing that
        board in a format suitable as input to our neural network.
        """
        parsed = []
        taken = board.get_taken()
        dead = board.get_dead()
        available = board.get_available_not_dead()

        # We get the tempo bit from the board, as well
        # as the total number of taken, dead, and available squares.
        input_TEMPO = [board.turn]#, len(taken), len(dead), len(available)]
        #input_DEAD = [int(len(dead) % 2 == 0)] # TEST: Just give the *parity* of *available* squares

        # I'm going to add to this the parity score for each row, then each
        # column, then finally the diagonal and antidiagonal.
        input_PAR = []
        input_PAR += [int(c % 2 == 0) for c in board.row_count]
        input_PAR += [int(c % 2 == 0) for c in board.col_count]
        input_PAR += [int(c % 2 == 0) for c in board.diag_count]

        # We then construct maps of taken squares on the board
        input_IMG = []
        input_C = []
        input_D = []
        for i in range(board.size):
            sub = []
            for j in range(board.size):
                if (i, j) in taken:
                    sub.append(1)
                else:
                    sub.append(0)
            input_IMG.append(sub)
                
                # if (i, j) in dead:
                #     input_C.append(1)
                # else:
                #     input_C.append(0)

                # if (i, j) in available:
                #     input_D.append(1)
                # else:
                #     input_D.append(0)

        parsed.append(input_IMG)
        parsed.append(input_TEMPO)
        #parsed.append(input_DEAD)
        parsed.append(input_PAR)
        return parsed

    def parseBoard_Best(self, board):
        """
        Takes in a board and returns a list representing that
        board in a format suitable as input to our neural network.
        """
        parsed = []
        taken = board.get_taken()
        dead = board.get_dead()
        
        # We get the tempo bit from the board, as well
        # as the total number of taken, dead, and available squares.
        input_A = [board.turn]
        input_B = [len(dead)]

        # I'm going to add to this the parity score for each row, then each
        # column, then finally the diagonal and antidiagonal.
        input_C = []
        input_C += [int(c % 2 == 0) for c in board.row_count]
        input_C += [int(c % 2 == 0) for c in board.col_count]
        input_C += [int(c % 2 == 0) for c in board.diag_count]

        # We then construct maps of taken, dead, and available squares
        # on the board.
        input_D = []
        for i in range(board.size):
            for j in range(board.size):
                if (i, j) in taken:
                    input_D.append(1)
                else:
                    input_D.append(0)
                
        parsed.append(input_A)
        parsed.append(input_B)
        parsed.append(input_C)
        parsed.append(input_D)
        return parsed

    def parseBoard_Better(self, board):
        """
        Takes in a board and returns a list representing that
        board in a format suitable as input to our neural network.
        """
        parsed = []
        taken = board.get_taken()
        dead = board.get_dead()
        
        # We get the tempo bit from the board, as well
        # as the total number of taken, dead, and available squares.
        input_A = [board.turn]
        input_B = [len(dead)]

        # I'm going to add to this the parity score for each row, then each
        # column, then finally the diagonal and antidiagonal.
        input_C = []
        input_C += [int(c % 2 == 0) for c in board.row_count]
        input_C += [int(c % 2 == 0) for c in board.col_count]
        input_C += [int(c % 2 == 0) for c in board.diag_count]

        # We then construct maps of taken, dead, and available squares
        # on the board.
        input_D = []
        input_E = []
        input_F = []
        for i in range(board.size):
            for j in range(board.size):
                # Collect rows
                if (i, j) in taken:
                    input_D.append(1)
                else:
                    input_D.append(0)

                # Collect columns
                if (j, i) in taken:
                    input_E.append(1)
                else:
                    input_E.append(0)

                # Collect diagonal
                if board.is_on_diagonal((i, j)):
                    if (i, j) in taken:
                        input_F.append(1)
                    else:
                        input_F.append(0)

        # Don't combine these for loops together; order of input_F matters!
        for i in range(board.size):
            for j in range(board.size):                
                # Collect antidiagonal
                if board.is_on_antidiagonal((i, j)):
                    if (i, j) in taken:
                        input_F.append(1)
                    else:
                        input_F.append(0)
                
        parsed.append(input_A)
        parsed.append(input_B)
        parsed.append(input_C)
        parsed.append(input_D)
        parsed.append(input_E)
        parsed.append(input_F)
        return parsed

    def train(self, x_train, y_train):
        self.model.compile(optimizer='adam',
            loss=self.loss_fn,
            metrics=['mean_squared_error']) # Note: accuracy is only a good metric for classification problems.
        self.model.fit(x_train, y_train, batch_size=self.parameters["batch size"], 
            epochs=self.parameters["num epochs"],
            shuffle=self.parameters["shuffle"],
            callbacks=[callback])

    def evaluate_board(self, board):
        # Fully-featured prediction
        # x_test = self.parseBoardAsInput(board)
        # input_A = x_test[0]
        # input_PAR = x_test[1]
        # input_B = x_test[2]
        # input_C = x_test[3]
        # input_D = x_test[4]
        # prediction = self.model.predict([
        #     numpy.asarray([input_A]),
        #     numpy.asarray([input_PAR]),
        #     numpy.asarray([input_B]),
        #     numpy.asarray([input_C]),
        #     numpy.asarray([input_D])
        # ])
        
        # Flat prediction
        x_test = self.parseBoard_Flat(board)
        prediction = self.model.predict(numpy.asarray([numpy.asarray(x_test)]))

        # Convolutional prediction
        if self.modelName == 'convolutional':
            x_test = self.parseBoard_Convolutional(board)
            linear_output = self.model.predict(numpy.asarray([numpy.asarray(x_test)])) # pi, v
            prediction = linear_output

        # Featured Convolutional Prediction
        elif self.modelName == 'featured_convolutional':
            x_test = self.parseBoard_Featured_Convolutional(board)
            input_IMG = x_test[0]
            input_TEMPO = x_test[1]
            #input_DEAD = x_test[2]
            input_PAR = x_test[2]
            prediction = self.model.predict([
                numpy.asarray([input_IMG]),
                numpy.asarray([input_TEMPO]),
                #numpy.asarray([input_DEAD]),
                numpy.asarray([input_PAR])
            ])

        # "Best" Prediction
        # x_test = self.parseBoard_Best(board)
        # input_A = x_test[0]
        # input_B = x_test[1]
        # input_C = x_test[2]
        # input_D = x_test[3]
        # prediction = self.model.predict([
        #     numpy.asarray([input_A]),
        #     numpy.asarray([input_B]),
        #     numpy.asarray([input_C]),
        #     numpy.asarray([input_D])
        # ])

        # "Better?" Prediction
        # x_test = self.parseBoard_Better(board)
        # input_A = x_test[0]
        # input_B = x_test[1]
        # input_C = x_test[2]
        # input_D = x_test[3]
        # input_E = x_test[4]
        # input_F = x_test[5]
        # prediction = self.model.predict([
        #     numpy.asarray([input_A]),
        #     numpy.asarray([input_B]),
        #     numpy.asarray([input_C]),
        #     numpy.asarray([input_D]),
        #     numpy.asarray([input_E]),
        #     numpy.asarray([input_F])
        # ])

        #print("Prediction:", prediction[0][0])
        return prediction[0][0]

    def next_move(self, board, nondeterminism=False):
        opn = board.get_available()
        dead = board.get_dead()
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
            cpy = copy.deepcopy(board)
            cpy.update(pos)

            value = self.evaluate_board(cpy)
            scores[pos] = value

        # print("Board scores:")
        # for pos, score in scores.items():
        #     print(f"{pos} - {score}")
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
