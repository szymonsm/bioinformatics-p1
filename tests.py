import unittest
import numpy as np
import pandas as pd
from src.algorithms import NeedlemanWunsch, SmithWaterman
import os

class TestNeedlemanWunsch(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.sequence1 = "ACGT"
        self.sequence2 = "AGT"
        self.substitution_matrix_file_path = os.path.join(os.path.dirname(__file__), 'data', 'substitution_matrix_test.csv')
        self.output_filepath = os.path.join(os.path.dirname(__file__), 'results', 'tests', 'logs-needleman-wunsch.txt')

    def test_matrix_initialization(self):
        # Initialize Needleman-Wunsch object
        nw = NeedlemanWunsch(n=1, sequence1=self.sequence1, sequence2=self.sequence2,
                             substitution_matrix_filepath=self.substitution_matrix_file_path, output_filepath=self.output_filepath)
        nw.init_matrix()

        # Check initial values in matrix
        self.assertEqual(nw.matrix.iloc[0, 0], 0)
        self.assertEqual(nw.matrix.iloc[0, 1], -6)
        self.assertEqual(nw.matrix.iloc[1, 0], -6)

    def test_matrix_filling(self):
        # Initialize Needleman-Wunsch object and fill the matrix
        nw = NeedlemanWunsch(n=1, sequence1=self.sequence1, sequence2=self.sequence2,
                             substitution_matrix_filepath=self.substitution_matrix_file_path, output_filepath=self.output_filepath)
        nw.init_matrix()
        nw.fill_matrix()

        # Verify filled matrix values match the expected Needleman-Wunsch calculations
        expected_value = 2  # A match between "A" and "A" at (1, 1)
        self.assertEqual(nw.matrix.iloc[1, 1], expected_value)

        expected_value = 1  # Mismatch between "C" and "G" at (2, 2) but match between "A" and "A" at (1, 1)
        self.assertEqual(nw.matrix.iloc[2, 2], expected_value)

        expected_value = 4  # Match between "T" and "T" at (4, 3) and match between "G" and "G" at (3, 2) with value 2
        self.assertEqual(nw.matrix.iloc[4, 3], expected_value)

    def test_traceback(self):
        # Initialize Needleman-Wunsch object, fill matrix, and run traceback
        nw = NeedlemanWunsch(n=10, sequence1=self.sequence1, sequence2=self.sequence2,
                             substitution_matrix_filepath=self.substitution_matrix_file_path, output_filepath=self.output_filepath)
        nw.init_matrix()
        nw.fill_matrix()
        print(nw.matrix)
        nw.traceback(save=False)

        # Check if the alignment is correct
        # expected_alignment1 = "ACGT"
        # expected_alignment2 = "A-GT"
        # Check if any of the tracebacks match the expected alignment
        print(nw.tracebacks)
        self.assertTrue(['up_left', 'up_left', 'up', 'up_left'] in nw.tracebacks)

    def test_multiple_tracebacks(self):
        # Initialize Needleman-Wunsch object for multiple alignments
        nw = NeedlemanWunsch(n=10, sequence1=self.sequence1, sequence2=self.sequence2,
                             substitution_matrix_filepath=self.substitution_matrix_file_path, output_filepath=self.output_filepath)
        nw.init_matrix()
        nw.fill_matrix()
        nw.traceback(save=False)

        # Check that we get two different alignments (if applicable)
        # I have manually checked that there is only one different alignment possible for this case
        self.assertEqual(len(nw.tracebacks), 1)


class TestSmithWaterman(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.sequence1 = "ACGT"
        self.sequence2 = "AGT"
        self.substitution_matrix_file_path = os.path.join(os.path.dirname(__file__), 'data', 'substitution_matrix_test.csv')
        self.output_filepath = os.path.join(os.path.dirname(__file__), 'results', 'tests', 'logs-smith-waterman.txt')

    def test_matrix_initialization(self):
        # Initialize Smith-Waterman object
        sw = SmithWaterman(n=1, sequence1=self.sequence1, sequence2=self.sequence2,
                           substitution_matrix_filepath=self.substitution_matrix_file_path, output_filepath=self.output_filepath)
        # sw.substitution_matrix = self.substitution_matrix
        sw.init_matrix()

        # Check initial values in matrix are all zeros
        self.assertTrue((sw.matrix.values == 0).all())

    def test_matrix_filling(self):
        # Initialize Smith-Waterman object and fill the matrix
        sw = SmithWaterman(n=1, sequence1=self.sequence1, sequence2=self.sequence2,
                           substitution_matrix_filepath=self.substitution_matrix_file_path, output_filepath=self.output_filepath)
        sw.init_matrix()
        sw.fill_matrix()

        # Check that the matrix is correctly filled with local alignment scores
        expected_value = 2  # Match between "A" and "A" at (1, 1)
        self.assertEqual(sw.matrix.iloc[1, 1], expected_value)

        expected_value = 4  # Match between "T" and "T" at (4, 3)
        self.assertEqual(sw.matrix.iloc[4, 3], expected_value)

        # Check that there are no negative values
        self.assertTrue((sw.matrix.values >= 0).all())

    def test_traceback(self):
        # Initialize Smith-Waterman object, fill matrix, and run traceback
        sw = SmithWaterman(n=10, sequence1=self.sequence1, sequence2=self.sequence2,
                           substitution_matrix_filepath=self.substitution_matrix_file_path, output_filepath=self.output_filepath)
        sw.init_matrix()
        sw.fill_matrix()
        sw.traceback(save=False)

        # Check if the alignment is correct
        # expected_alignment1 = "GT"
        # expected_alignment2 = "GT"
        # Check if any of the tracebacks match the expected alignment
        self.assertTrue(['up_left', 'up_left'] in sw.all_tracebacks)

    def test_traceback_with_multiple_starting_points(self):
        # Initialize Smith-Waterman object for multiple tracebacks
        sw = SmithWaterman(n=10, sequence1=self.sequence1, sequence2=self.sequence2,
                           substitution_matrix_filepath=self.substitution_matrix_file_path, output_filepath=self.output_filepath)
        sw.init_matrix()
        sw.fill_matrix()
        sw.traceback(save=False)

        # Check that the traceback starts from the maximum score points
        max_score = sw.matrix.max().max()
        max_indices = np.argwhere(sw.matrix.values == max_score)

        # Check if max indices are (4, 3)
        self.assertTrue((4, 3) in max_indices)

        # Check that there is only 1 traceback
        # I have manually checked that there is only one different alignment possible for this case
        self.assertEqual(len(sw.all_tracebacks), 1)



# Run the tests
if __name__ == '__main__':
    unittest.main()