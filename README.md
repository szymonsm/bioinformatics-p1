# How to check the solution?
1. Install requirements for the task:
  ```
  pip install requirements.txt
  ```
2. sample.ipynb - this notebook contains sample usage of the two algorithms: Needleman-Wunsch and Smith-Waterman.
3. tests.py - unit tests to check if everything goes well. Run:
   ```
   python tests.py
   ```
4. /data - folder with 2 substitution matrices: substitution_matrix.csv - the matrix used in sample.ipynb, substitution_matrix_test.csv - the matrix used in the unit tests.
5. /results - folder with *.txt files corresponding to the detected alignements from sample.ipynb file.
6. /src - contains the algorithms.py file with NeedlemanWunsch and SmithWaterman classes.
