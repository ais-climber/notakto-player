# Class for Notakto game board
import copy

class Board:
    def __init__(self, size):
        self.size = size
        self.game_board = [["_" for i in range(size)] for i in range(size)]
        self.turn = 0

        # For potential use, track the number of "X"'s on the board.
        self.xcount = 0

        # For internal use, track the number of X's total on the board,
        # the number of X's for each row, the number of X's for each column,
        # and the number of X's on the diagonals.
        self.xcount = 0
        self.row_count = [0 for i in range(size)] 
        self.col_count = [0 for j in range(size)]
        self.diag_count = [0, 0]

    def get(self, i, j):
        return self.game_board[i][j]

    def get_available(self):
        """
        Returns a list of coordinates of all available (empty) squares.
        """
        l = []

        for i in range(self.size):
            for j in range(self.size):
                if self.game_board[i][j] == "_":
                    l.append((i, j))
        
        return l

    def get_available_not_dead(self):
        """
        Returns a list of coordinates of all available (empty) squares
        that are also not dead.
        """
        l = []

        for i in range(self.size):
            for j in range(self.size):
                if self.game_board[i][j] == "_":
                    l.append((i, j))
        
        return [p for p in l if p not in self.get_dead()]

    def get_taken(self):
        """
        Returns a list of coordinates of all unavailable (taken) squares.
        """
        l = []

        for i in range(self.size):
            for j in range(self.size):
                if self.game_board[i][j] == "X":
                    l.append((i, j))
        
        return l

    def get_dead(self):
        """
        Returns a list of coordinates of "dead" squares, i.e. those squares
        such that placing an 'X' there would end the game.
        """
        l = []
        near_end = self.size - 1
        
        for i in range(self.size):
            for j in range(self.size):
                if not self.is_already_taken((i, j)):
                    if self.row_count[i] == near_end or self.col_count[j] == near_end:
                        l.append((i, j))

                    elif self.is_on_diagonal((i, j)) and self.diag_count[0] == near_end:
                        l.append((i, j))

                    elif self.is_on_antidiagonal((i, j)) and self.diag_count[1] == near_end:
                        l.append((i, j))

        return l

    def is_on_diagonal(self, p):
        """
        Returns True iff given coordinate point is on the main diagonal.
        """
        i, j = p
        return i == j

    def is_on_antidiagonal(self, p):
        """
        Returns True iff given coordinate point is on the antidiagonal.
        """
        i, j = p
        return i + j == self.size - 1

    def is_already_taken(self, p):
        """
        Returns True iff given coordinate point is already marked 'X'
        """
        i, j = p
        return self.game_board[i][j] == "X"

    def update(self, p):
        i, j = p
        if self.game_board[i][j] != "X":
            self.game_board[i][j] = "X"

            # Update internal counts
            self.xcount += 1
            self.row_count[i] += 1
            self.col_count[j] += 1

            if self.is_on_diagonal((i, j)):
                self.diag_count[0] += 1
            if self.is_on_antidiagonal((i, j)):
                self.diag_count[1] += 1

            # Pass board off to next player (update whose turn it is)
            self.update_turn()

    def update_turn(self):
        self.turn = 1 - self.turn

    def game_over(self):
        """
        Function to determine whether the game is over.

        >>> a = Board(2)
        >>> a.update((0, 0))
        >>> a.game_over()
        False
       
        >>> b = Board(2)
        >>> b.update((0, 0))
        >>> b.update((0, 1))
        >>> b.game_over()
        True
        
        >>> c = Board(2)
        >>> c.update((0, 0))
        >>> c.update((1, 1))
        >>> c.game_over()
        True
        """
        for i in range(self.size):
            if self.row_count[i] == self.size:
                return True
            if self.col_count[i] == self.size:
                return True
        if self.diag_count[0] == self.size or self.diag_count[1] == self.size:
            return True
        
        # Otherwise
        return False

    def play_until_end(self, next_move):
        """
        Starting at the given board, we play
        the game with strategy 'next_move' until the game
        is finished.  We return 0 or 1 to indicate the winner.

        >>> 
        """
        b = copy.deepcopy(self)

        while not b.game_over():
            p = next_move(b)
            b.update(p)
            
        # Return the winner
        if b.turn == 0:
            return 0
        elif b.turn == 1:
            return 1
    
    def MCTS(self, next_move):
        """
        A function to perform Monte Carlo Tree Search.

        TODO: Description

        We return the average scores of the explored
        branches.
        """
        b = copy.deepcopy(self)
        roots = []
        winners = []

        # We explore the game tree up to two more
        # steps, so long as the game doesn't end.
        # We store these game boards in 'roots'.
        # TODO:
        roots = []
        preroots = []

        opn = b.get_available()
        dead = b.get_dead()
        choices = [c for c in opn if c not in dead]
        if choices != []:

            # Make the available moves, and add these to our roots.
            for pos in choices:
                cpy = copy.deepcopy(b)
                cpy.update(pos)
                preroots.append(cpy)

        # For each of our roots, we now explore one more step.
        if preroots != []:
            for r in preroots:            
                opn = r.get_available()
                dead = r.get_dead()
                choices = [c for c in opn if c not in dead]
                if choices != []:
                    for pos in choices:
                        cpy = copy.deepcopy(r)
                        cpy.update(pos)
                        roots.append(cpy)
        else:
            roots = [b]
        
        if roots == []:
            roots = [b]

        # For each root game board, we play against
        # ourself until the game is finished, and get
        # the winner.
        for r in roots:
            winners.append(r.play_until_end(next_move))

        # Finally, we average the scores of these games,
        # and return the move with the best score.
        # Note that if the current player is:
        #     0 - a lower score is better (most 0 wins)
        #     1 - a higher score is better (most 1 wins)
        # So we invert the score if the current player is '0'.
        avg = sum(winners) / len(winners)
        if self.turn == 0:
            return 1 - avg
        else:
            return avg


    def __eq__(self, someBoard):
        if self.size != someBoard.size:
            return False
        
        for i in range(self.size):
            for j in range(self.size):
                if self.game_board[i][j] != someBoard.game_board[i][j]:
                    return False
        
        if self.turn != someBoard.turn:
            return False
        
        # Otherwise
        return True

    def __list__(self):
        """
        List representation of the board flattens
        it, in order to be fed into, e.g., a neural network.
        """
        l = []
        for i in self.game_board:
            l += self.game_board[i]
        
        return l

    def __str__(self):
        s = str(self.turn) + "\n"

        for i in range(self.size):
            for j in range(self.size):
                s += self.game_board[i][j]
            s += "\n"
        
        s.rstrip("\n")
        return s
        
if __name__ == "__main__":
    # Test a game board
    # print("Example board...")
    # b = Board(4)
    # b.update(0, 0)
    # b.update(0, 1)
    # b.update(1, 1)
    # b.update(3, 0)
    # b.update(2, 1)
    # b.update(1, 2)
    # #b.update(0, 3)
    # print(b)
    # print(f"Game finished? : {b.game_over()}")

    import doctest
    doctest.testmod()
    

    

    

