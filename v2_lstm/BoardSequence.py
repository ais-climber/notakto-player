from Board import *

class BoardSequence:

    def __init__(self, initial_board):
        self.boardlist = [initial_board]
        self.size = initial_board.size

        self.initial_board = initial_board
        self.current_board = initial_board

    def append(self, new_board):
        self.boardlist.append(new_board)
        self.current_board = new_board

    def at(self, i):
        return self.boardlist[i]

    def truncate(self, i):
        """
        Returns a new BoardSequence object that
        consists of all of the boards up to (and including)
        position 'i'.
        """
        seq = BoardSequence(self.initial_board)

        for j in range(len(self.boardlist)):
            if j <= i:
                seq.append(self.boardlist[j])

        return seq

    def evaluate(self, next_move):
        """
        A function to evaluate this sequence of boards.

        The basic idea is this:  In chess, the losing player often
        revisits a board to analyze what went wrong -- often precisely
        *when* they lost.  A chess board is bad not only because it resulted
        in a loss, but also (and mainly) because it would have resulted in
        a loss *many moves ago*.

        Notakto is similar, and for misere games in general this is especially
        true.  A player must play *perfectly*, and should especially avoid
        any boards where _not only have they lost_, but they _lost long ago_.

        We take the boards in the sequence and weigh them by (1/(2**i)) (starting
        from the current move, i.e. the end of the sequence).  We evaluate each
        of these boards, and sum up the weighted result to get the total score.
        """
        # Weights for the early game -- only look one step back
        weights = [1 if i==1 else 0 for i in range(len(self.boardlist), 0, -1)]

        # Weights for later game -- look 4 steps back, with diminishing value.
        if len(self.boardlist) >= 5:
            weights[-1] = 0.70
            weights[-2] = 0.15
            weights[-3] = 0.075
            weights[-4] = 0.075

        score = 0
        for i in range(len(self.boardlist)):
            # We have to pass in this board sequence in order to progress the game via next_move
            score += weights[i]*self.at(i).evaluate(next_move)
        return score

    def __eq__(self, seq):
        if self.initial_board != seq.initial_board:
            return False
        if self.current_board != seq.current_board:
            return False
        if len(self.boardlist) != len(seq.boardlist):
            return False
        
        for i in range(len(self.boardlist)):
            if self.at(i) != seq.at(i):
                return False
        
        return True




    