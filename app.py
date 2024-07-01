#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 08:58:03 2024

@author: ruvinjagoda
"""

import chess
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import chess.engine

app = Flask(__name__)

CORS(app)

# Global variables
engine = None
model = None
val_model = None

# Dictionary to map the board positions
board_positions = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7
}
# Path to the Stockfish engine
engine_path = "./stockfish-windows-x86-64-vnni512.exe"
NUMOFMOVES = 2


# Function to convert square to index
def square_to_index(square):
    letter = chess.square_name(square)  # Get algebraic notation of the square
    row = 8 - int(letter[1])  # Convert rank to row index
    # Map file to column index using board_positions dictionary
    column = board_positions[letter[0]]
    return row, column  # Return row and column indices


# Function to convert board to matrix representation
def board_to_matrix(board):
    # Initialize a 3D numpy array to represent the board
    # 12 dimensions -> different chess piece types (Pawn, Knight, Bishop, Rook, Queen, King) for white and black
    # +2 dimensions -> legal moves for current player and opponent
    board_3d = np.zeros((14, 8, 8), dtype=np.int8)

    # Iterate over each piece type
    for piece in chess.PIECE_TYPES:
        # Iterate over each white piece on the board
        for square in board.pieces(piece, chess.WHITE):
            # Convert square index to row and column indices and mark the corresponding position in the matrix with 1
            index = np.unravel_index(square, (8, 8))
            # piece - 1 -> layer of 3D array corresponding to piece type (0-5)
            # 7 - index[0] -> row index (row numbering in chess module is reversed)
            # index[1] -> column index
            board_3d[piece - 1][7 - index[0]][index[1]] = 1

        # Iterate over each black piece on the board
        for square in board.pieces(piece, chess.BLACK):
            # Convert square index to row and column indices, and mark the corresponding position in the matrix with 1
            index = np.unravel_index(square, (8, 8))
            # piece + 5 -> layer of 3D array corresponding to black piece type (6-11)
            # 7 - index[0] -> row index (row numbering in chess module is reversed)
            # index[1] -> column index
            board_3d[piece + 5][7 - index[0]][index[1]] = 1

    # Store legal moves for the current player
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_3d[12][i][j] = 1  # Layer 12

    # Store legal moves for the opponent player
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_3d[13][i][j] = 1  # Layer 13

    # Restore the original player turn
    board.turn = aux

    return board_3d  # Return the 3D matrix representati


def move_to_grid(move):
    from_square = move.from_square
    to_square = move.to_square

    # Initialize an empty 8x8 grid
    grid = np.zeros((8, 8), dtype=np.int8)

    # Convert the from_square and to_square to row, col format
    from_row, from_col = divmod(from_square, 8)
    to_row, to_col = divmod(to_square, 8)

    # Encode the move
    grid[from_row, from_col] = 1  # Starting position
    grid[to_row, to_col] = 2      # Ending position
    return grid



def analyze_position(board_fen, engine, num_moves=NUMOFMOVES):
    try:
        board = chess.Board(board_fen)
        
        # Get top moves with multipv
        result = engine.analyse(board, chess.engine.Limit(depth=15), multipv=num_moves)
        
        # Extract the best moves and their principal variations (PVs)
        top_moves = []
        for move_info in result:
            move = move_info["pv"][0].uci()
            score = move_info["score"]

            # Determine centipawn and mate values
            centipawns = score.relative.score() if not score.is_mate() else 0
            mate = score.relative.mate() if score.is_mate() else 0

            top_moves.append({
                "move": move,
                "centipawns": centipawns,
                "mate": mate
            })
        
        # Generate move sequences
        move_sequence = [move.uci() for move in result[0]["pv"][:num_moves]]
        
        # Extract moves, centipawns, and mates from top_moves
        moves = [move["move"] for move in top_moves]
        centipawns = [move["centipawns"] for move in top_moves]
        mates = [move["mate"] for move in top_moves]

        return moves, moves[0], centipawns, mates, move_sequence
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], None, [], [], []


# Convert move to label
def move_to_label(move):
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square


# Function to get the best move
def get_best_move_from_prediction(prediction):
    flat_index = np.argmax(prediction)
    from_square = flat_index // 64
    to_square = flat_index % 64
    move = chess.Move(from_square, to_square)
    return move


def standardizeCenti(data):
    data = np.array(data)

    mean = -24.645909333333332
    std_dev = 191.5498703657434

    standardized_data = (data - mean) / std_dev
    print(np.array(standardized_data))
    return np.array(standardized_data).reshape(1, 2)


def standardizeMate(data):
    data = np.array(data)
    mean = -0.007666
    std_dev = 0.47675210096792786

    standardized_data = (data - mean) / std_dev
    return np.array(standardized_data).reshape(1, 2)

# Get the AI's move


def get_ai_move(board, engine):

    feature_board = board_to_matrix(board)
    feature_board = np.array(feature_board)

    try:
        
        top_moves, best_move, centipawns, mates, move_sequence = analyze_position(
            board.fen(), engine)
        
        best_move_onehot = tf.one_hot(move_to_label(
            chess.Move.from_uci(best_move)), depth=4096, dtype=tf.uint8)
        centipawns = standardizeCenti(centipawns)
        mates = standardizeMate(mates)
        labels = [tf.one_hot(move_to_label(chess.Move.from_uci(
            move)), depth=4096, dtype=tf.uint8) for move in top_moves]
        while len(labels) < 2:
            labels.append(np.zeros((4096), dtype=np.uint8))

        matrix_top_moves = np.array(labels)

        labels = [tf.one_hot(move_to_label(chess.Move.from_uci(
            move)), depth=4096, dtype=tf.uint8) for move in move_sequence]
        while len(labels) < 2:
            labels.append(np.zeros((4096), dtype=np.uint8))

        matrix_sequence_moves = np.array(labels)
        # Reshape matrix_top_moves
        # Reshape matrix_top_moves and other arrays
        matrix_top_moves_reshaped = matrix_top_moves.reshape(-1, 2, 4096)
        matrix_sequence_moves_reshaped = matrix_sequence_moves.reshape(
            -1, 2, 4096)
        feature_board = feature_board.reshape(-1, 14, 8, 8)
        centipawns = np.array(centipawns).reshape(1, 2)
        mates = np.array(mates).reshape(1, 2)
        best_move_onehot = best_move_onehot.numpy().reshape(1, 4096)

        predictions = model.predict(
            [feature_board, centipawns, mates, matrix_sequence_moves_reshaped, matrix_top_moves_reshaped, best_move_onehot])
        best_move_model = get_best_move_from_prediction(predictions[0])
        print(
            f"The best move predicted by the model is: {best_move_model.uci()}")
        top_moves = [move for move in top_moves]
        print(f"The top moves by Sf : {top_moves}")
        print(f"The best move predicted by the stockfish is: {best_move}")
        if (best_move_model in list(board.legal_moves)):
            return best_move_model.uci()
        else:
            print(
                f"Predicted move not in legal moves {best_move.uci()}. Using Stockfish to get the best move.")
            return best_move

    except Exception as e:
        print(f"An error occurred: {e}. Using Stockfish to get the best move.")
        return best_move

# Function to initialize the Stockfish engine
def init_engine():
    global engine
    # Initialize the Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    # Set Elo level
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 3000})
    print("Engine init done")

# Function to load the machine learning models
def load_models():
    global model, val_model
    # Load models
    model = load_model("./model_update_acc54_2.h5")
    val_model = load_model('./chess_model_update.h5')
    print("Models loaded")

# Initialize engine and models when the Flask app starts
init_engine()
load_models()



@app.route('/get_move', methods=['POST'])
def get_move():
    board_state = request.json['board']
    board = chess.Board(board_state)
    move = get_ai_move(board, engine)
    return jsonify({'move': move})


@app.route('/get_evaluation', methods=['POST'])
def get_evaluation():
    board_state = request.json['board']
    board = chess.Board(board_state)
    board_matrix = board_to_matrix(board)
    # Ensure input is in the correct shape
    board_matrix = np.expand_dims(board_matrix, axis=0)

    # Make a prediction
    evaluation = val_model.predict(board_matrix)

    # Print the evaluation
    print("Evaluation:", evaluation)
    return jsonify({'evaluation': float(evaluation[0][0])})


if __name__ == '__main__':

     app.run(host='0.0.0.0', debug=True)
