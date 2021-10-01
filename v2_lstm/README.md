Version 2, using an LSTM

Idea:  If we want to teach a neural network a winning strategy for a game like Notakto, it's not enough
to provide it with instances of game boards and have it learn how to evaluate these.  This is because a
winning strategy demands more from the system:  It requires that the system play _perfectly_.  In particular,
in Notakto perfect play involves perfect moves _in response to the opponent's move_.

We cannot teach a flat network how to respond to moves, since it has no memory of the previous boards
in the sequence.  But this is exactly what LSTMs (and other recurrent nets) were designed for.  The plan
is as follows:
1. Generate game board _sequences_, and tag these whole sequences with sequence-dependent evaluations.
2. Train an LSTM with a linear output node on these game sequences, have it predict the evaluation score.
3. Given a new board B arising from sequence S, ask the LSTM to evaluate S (which will be the evaluation for B).
4. Have our system play based on its best-evaluated boards.
5. Profit

File structure:
- Board.py
    - Class representing a single board
- BoardSequence.py
    - Class representing a whole sequence of boards.  Includes a named 'current board'
- Nets.py
    - Class containing LSTM + linear activation net, along with a 'next_move' function for taking a next move
    - also includes a control group net to test against
- strategies.py
    - A module containing a number of strategies for playing Notakto
- generate_games.py
    - A module for generating sequences of boards via greedy play, and simultaneously tagging them with evaluataions
      (in the past these were done seperately, but it was really expensive)
- heatmap.py
    - A module for generating heatmaps of a player's behavior
- main.py
    - The main module, for doing the training and setting up play.


