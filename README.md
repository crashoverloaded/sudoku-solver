# sudoku-solver

- **models_and_puzzle/models/sudokunet.py**  Holds the SudokuNet CNN architecture implemented with TensorFlow and Keras
- **models_and_puzzle/sudoku/puzzle.py**  Contains two helper utilities for finding the sudoku puzzle board itself as well as digits therein.

- As with all CNNs, SudokuNet needs to be trained with data. Our **train_digit_classifier.py** script will train a digit OCR model on the MNIST dataset.

- Once SudokuNet is successfully trained, we’ll deploy it with our **solve_sudoku_puzzle.py** script to solve a sudoku puzzle.
