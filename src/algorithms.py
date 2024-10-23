import numpy as np
import pandas as pd

class NeedlemanWunsch:
    '''
    Needleman-Wunsch algorithm for global sequence alignment.

    Parameters:
    n: int
        Number of alignments to output.
    sequence1: str
        First sequence to align.
    sequence2: str
        Second sequence to align.
    substitution_matrix_filepath: str
        Filepath to the substitution matrix.
    output_filepath: str
        Filepath to save the alignments.
    gap_penalty: int
        Penalty for introducing a gap in the alignment. Default is -2.

    Methods:
    init_matrix()
        Initializes the alignment matrix.
    fill_matrix()
        Fills the alignment matrix.
    print_matrix()
        Prints the alignment matrix.
    traceback(save: bool = True, save_tracebacks: bool = False)
        Performs traceback to get the alignments.
    '''

    def __init__(self, n: int, sequence1: str, sequence2: str, substitution_matrix_filepath: str, output_filepath: str, gap_penalty: int = -2):
        self.n = n
        self.sequence1 = sequence1.upper()
        self.sequence2 = sequence2.upper()
        self.substitution_matrix = pd.read_csv(substitution_matrix_filepath, index_col=0)
        self.substitution_matrix.columns = [c.upper() for c in self.substitution_matrix.columns]
        self.substitution_matrix.index = self.substitution_matrix.columns
        self.gap_penalty = gap_penalty
        self.output_filepath = output_filepath
        self.matrix = None
        self.tracebacks = None

    def init_matrix(self):
        self.matrix = np.zeros((len(self.sequence1) + 1, len(self.sequence2) + 1))
        self.matrix[0] = np.arange(0, -6 * len(self.sequence2) - 1, -6)
        self.matrix[:, 0] = np.arange(0, -6 * len(self.sequence1) - 1, -6)
        self.matrix = pd.DataFrame(self.matrix)
        self.matrix.columns = [''] + [c for c in self.sequence2]
        self.matrix.index = [''] + [c for c in self.sequence1]

    def fill_matrix(self):

        if self.matrix is None:
            raise ValueError('Matrix not initialized.')
        
        for i in range(1, len(self.sequence1) + 1):
            for j in range(1, len(self.sequence2) + 1):
                # Get index as gen1
                gen1 = self.sequence1[i - 1]
                # Get index as gen2
                gen2 = self.sequence2[j - 1]

                # Get the score of the substitution
                _score1 = self.matrix.iloc[i - 1, j - 1] + self.substitution_matrix.loc[gen1, gen2]
                _score2 = self.matrix.iloc[i - 1, j] + self.gap_penalty
                _score3 = self.matrix.iloc[i, j - 1] + self.gap_penalty

                self.matrix.iloc[i, j] = max(_score1, _score2, _score3)

    def print_matrix(self):
        if self.matrix is None:
            raise ValueError('Matrix not initialized.')
        print(self.matrix)

    def _find_traceback_steps(self, x, y, path, paths):
        # If we've reached the top-left corner, add the path to paths
        if x == 0 and y == 0:
            paths.append(path[:])
            return
        
        if len(paths) == self.n:
            return

        potential_moves = []

        if x > 0 and y > 0:
            potential_moves.append('up_left')
        if x > 0:
            potential_moves.append('left')
        if y > 0:
            potential_moves.append('up')

        scores = {}

        for potential_move in potential_moves:
            if potential_move == 'up_left':
                gen1 = self.sequence1[x - 1]
                gen2 = self.sequence2[y - 1]
                scores['up_left'] = self.matrix.iloc[x - 1, y - 1] + self.substitution_matrix.loc[gen1, gen2]
            if potential_move == 'left':
                scores['left'] = self.matrix.iloc[x - 1, y] + self.gap_penalty
            if potential_move == 'up':
                scores['up'] = self.matrix.iloc[x, y - 1] + self.gap_penalty
        
        max_score = max(scores.values())

        for move, score in scores.items():
            if np.isclose(score, max_score):
                path.append(move)
                if move == 'up_left':
                    self._find_traceback_steps(x - 1, y - 1, path, paths)
                elif move == 'left':
                    self._find_traceback_steps(x - 1, y, path, paths)
                elif move == 'up':
                    self._find_traceback_steps(x, y - 1, path, paths)
                path.pop()

    def _get_all_tracebacks(self):
        paths = []
        # Start traceback from the bottom-right corner and trace back to (0, 0)
        self._find_traceback_steps(len(self.sequence1), len(self.sequence2), [], paths)
        return paths

    def traceback(self, save: bool = True, save_tracebacks: bool = False):

        if self.matrix is None:
            raise ValueError('Matrix not initialized.')
        
        self.tracebacks = self._get_all_tracebacks()
        
        for i, path in enumerate(self.tracebacks):
            x, y = len(self.sequence1), len(self.sequence2)
            aligned_seq1 = ''
            aligned_seq2 = ''
            for move in path:  # Reverse the path to build the alignment from start to finish
                if move == 'up_left':
                    aligned_seq1 = self.sequence1[x - 1] + aligned_seq1
                    aligned_seq2 = self.sequence2[y - 1] + aligned_seq2
                    x -= 1
                    y -= 1
                elif move == 'left':
                    aligned_seq1 = '-' + aligned_seq1
                    aligned_seq2 = self.sequence2[y - 1] + aligned_seq2
                    x -= 1
                elif move == 'up':
                    aligned_seq1 = self.sequence1[x - 1] + aligned_seq1
                    aligned_seq2 = '-' + aligned_seq2
                    y -= 1

            print(f'Global alignment no. {i + 1}:')
            print(aligned_seq1)
            print(aligned_seq2)
            print(f"Score: {self.matrix.iloc[len(self.sequence1), len(self.sequence2)]}")
            print()

            if save:
                with open(self.output_filepath, 'a') as f:
                    f.write(f'Global alignment no. {i + 1}:\n')
                    f.write(aligned_seq1 + '\n')
                    f.write(aligned_seq2 + '\n')
                    if save_tracebacks:
                        f.write(f"Traceback: {path}\n")
                    f.write(f"Score: {self.matrix.iloc[len(self.sequence1), len(self.sequence2)]}\n\n")


# Smith-Waterman algorithm for local sequence alignment (take from upper class)
class SmithWaterman(NeedlemanWunsch):
    '''
    Smith-Waterman algorithm for local sequence alignment.

    Parameters:
    n: int
        Number of alignments to output.
    sequence1: str
        First sequence to align.
    sequence2: str
        Second sequence to align.
    substitution_matrix_filepath: str
        Filepath to the substitution matrix.
    output_filepath: str
        Filepath to save the alignments.
    gap_penalty: int
        Penalty for introducing a gap in the alignment. Default is -2.

    Methods:
    init_matrix()
        Initializes the alignment matrix.
    fill_matrix()
        Fills the alignment matrix.
    print_matrix()
        Prints the alignment matrix.
    traceback(save: bool = True, save_tracebacks: bool = False)
        Performs traceback to get the alignments.
    '''

    # Overwrite the init_matrix method to initialize the matrix with zeros
    def init_matrix(self):
        self.matrix = pd.DataFrame(np.zeros((len(self.sequence1) + 1, len(self.sequence2) + 1)))
        self.matrix.columns = [''] + [c for c in self.sequence2]
        self.matrix.index = [''] + [c for c in self.sequence1]
        

    def fill_matrix(self):

        if self.matrix is None:
            raise ValueError('Matrix not initialized.')
        
        for i in range(1, len(self.sequence1) + 1):
            for j in range(1, len(self.sequence2) + 1):
                # Get index as gen1
                gen1 = self.sequence1[i - 1]
                # Get index as gen2
                gen2 = self.sequence2[j - 1]

                # Get the score of the substitution
                _score1 = self.matrix.iloc[i - 1, j - 1] + self.substitution_matrix.loc[gen1, gen2]
                _score2 = self.matrix.iloc[i - 1, j] + self.gap_penalty
                _score3 = self.matrix.iloc[i, j - 1] + self.gap_penalty
                self.matrix.iloc[i, j] = max(0, _score1, _score2, _score3)

    def _get_all_tracebacks(self):
        paths = []
        indices = []
        # Find the maximum value in the matrix
        max_score = self.matrix.max().max()
        # Get the indices of the maximum value
        max_indices = np.argwhere(self.matrix.values == max_score)
        for max_index in max_indices:
            x, y = max_index
            self._find_traceback_steps(x, y, [], paths)
            indices.append((x, y))
        return paths, indices
    
    def _find_traceback_steps(self, x, y, path, paths):
        # Add if the current score is 0
        if self.matrix.iloc[x, y] == 0:
            paths.append(path[:])
            return
        return super()._find_traceback_steps(x, y, path, paths)
    
    def traceback(self, save: bool = True, save_tracebacks: bool = False):
            
            if self.matrix is None:
                raise ValueError('Matrix not initialized.')
            
            self.all_tracebacks, indices = self._get_all_tracebacks()
            
            for i, (path, (x_start, y_start)) in enumerate(zip(self.all_tracebacks, indices)):
                x = x_start
                y = y_start
                
                aligned_seq1 = ''
                aligned_seq2 = ''
                for move in path:
                    if move == 'up_left':
                        aligned_seq1 = self.sequence1[x - 1] + aligned_seq1
                        aligned_seq2 = self.sequence2[y - 1] + aligned_seq2
                        x -= 1
                        y -= 1
                    elif move == 'left':
                        aligned_seq1 = '-' + aligned_seq1
                        aligned_seq2 = self.sequence2[y - 1] + aligned_seq2
                        x -= 1
                    elif move == 'up':
                        aligned_seq1 = self.sequence1[x - 1] + aligned_seq1
                        aligned_seq2 = '-' + aligned_seq2
                        y -= 1

                print(f'Local alignment no. {i + 1}:')
                print(aligned_seq1)
                print(aligned_seq2)
                print(f"Score: {self.matrix.iloc[x_start, y_start]}")
                print()

                if save:
                    with open(self.output_filepath, 'a') as f:
                        f.write(f'Local alignment no. {i + 1}:\n')
                        f.write(aligned_seq1 + '\n')
                        f.write(aligned_seq2 + '\n')
                        if save_tracebacks:
                            f.write(f"Starting indices: {x_start}, {y_start}\n")
                            f.write(f"Traceback: {path}\n")
                        f.write(f"Score: {self.matrix.iloc[x_start, y_start]}\n\n")