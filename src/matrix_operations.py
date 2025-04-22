# src/matrix_operations.py
import torch

class MatrixOperations:
    @staticmethod
    def create_matrix(rows, cols, data=None, device='cpu'):
        """Create a matrix (Section 2.2).
        
        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.
            data (list, optional): Matrix entries as list of lists. If None, random matrix.
            device (str): 'cpu' or 'cuda'. Defaults to 'cpu'.

        Returns:
            torch.Tensor: Matrix of shape (rows, cols).
        
        Example:
            data=[[1, 2], [3, 4]] creates [[1, 2], [3, 4]].
            data=None with rows=2, cols=2 creates [[0.123, 0.456], [0.789, 0.234]] (random).
        """
        if data is None:
            # torch.rand: Creates tensor with random values in [0, 1).
            # Intent: Generate random matrix, e.g., [[0.5, 0.7], [0.2, 0.9]], for testing.
            # Purpose: Fast way to make a matrix without manual entries, supports CPU/GPU.
            return torch.rand(rows, cols, device=device)
        # torch.tensor: Converts list to tensor, e.g., [[1, 2], [3, 4]] -> tensor([[1., 2.], [3., 4.]]).
        # Intent: Use user-defined values for specific matrix operations.
        # Purpose: Creates tensor with float32 precision for math operations.
        return torch.tensor(data, dtype=torch.float32, device=device)

    @staticmethod
    def add_matrices(A, B):
        """Matrix addition (Section 2.2).
        
        Args:
            A (torch.Tensor): First matrix.
            B (torch.Tensor): Second matrix, same shape as A.

        Returns:
            torch.Tensor: A + B.

        Raises:
            ValueError: If shapes differ.

        Example:
            A=[[1, 2], [3, 4]], B=[[5, 6], [7, 8]] -> [[6, 8], [10, 12]].
            Equation: C[i,j] = A[i,j] + B[i,j].
        """
        if A.shape != B.shape:
            raise ValueError("Matrices must have the same dimensions.")
        # Operator +: Adds tensors element-wise, e.g., [[1, 2], [3, 4]] + [[5, 6], [7, 8]] = [[6, 8], [10, 12]].
        # Intent: Compute matrix sum as per linear algebra.
        # Purpose: Fast element-wise addition, optimized for CPU/GPU.
        return A + B

    @staticmethod
    def matrix_multiply(A, B):
        """Matrix multiplication (Section 2.2).
        
        Args:
            A (torch.Tensor): Matrix of shape (m, p).
            B (torch.Tensor): Matrix of shape (p, n).

        Returns:
            torch.Tensor: AB, shape (m, n).

        Raises:
            ValueError: If A.cols != B.rows.

        Example:
            A=[[1, 2], [3, 4]], B=[[5, 6], [7, 8]] -> [[19, 22], [43, 50]].
            Equation: C[i,j] = sum(A[i,k] * B[k,j]).
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix A's columns must equal Matrix B's rows.")
        # torch.matmul: Computes matrix product, e.g., [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]].
        # Intent: Perform matrix multiplication for linear transformations.
        # Purpose: Efficiently calculates AB, critical for neural networks.
        return torch.matmul(A, B)

    @staticmethod
    def solve_linear_system(A, b):
        """Solve Ax = b (Section 2.1).
        
        Args:
            A (torch.Tensor): Square matrix (n, n).
            b (torch.Tensor): Vector (n,).

        Returns:
            torch.Tensor: Solution x.

        Raises:
            ValueError: If A.rows != b.length.
            RuntimeError: If A is singular.

        Example:
            A=[[2, 1], [1, 3]], b=[5, 4] -> x=[1.857, 1.286].
            Equation: Ax = b, solve for x.
        """
        if A.shape[0] != b.shape[0]:
            raise ValueError("Matrix A rows must equal vector b length.")
        # torch.linalg.solve: Solves Ax = b, e.g., [[2, 1], [1, 3]]x = [5, 4] -> x=[1.857, 1.286].
        # Intent: Find x satisfying the linear system.
        # Purpose: Uses LU decomposition for fast, stable solution.
        return torch.linalg.solve(A, b)

    @staticmethod
    def transpose(A):
        """Matrix transpose (Section 2.2).
        
        Args:
            A (torch.Tensor): Matrix (m, n).

        Returns:
            torch.Tensor: A^T, shape (n, m).

        Example:
            A=[[1, 2], [3, 4]] -> [[1, 3], [2, 4]].
            Equation: (A^T)[i,j] = A[j,i].
        """
        # torch.t: Transposes 2D tensor, e.g., [[1, 2], [3, 4]] -> [[1, 3], [2, 4]].
        # Intent: Swap rows and columns for transpose operation.
        # Purpose: Fast transpose for linear algebra tasks.
        return torch.t(A)

    @staticmethod
    def determinant(A):
        """Determinant of square matrix (Section 4.1, referenced in 2.3).
        
        Args:
            A (torch.Tensor): Square matrix (n, n).

        Returns:
            torch.Tensor: Determinant (scalar).

        Raises:
            ValueError: If A is not square.

        Example:
            A=[[4, 7], [2, 6]] -> det=4*6 - 7*2=10.
            Equation: det(A) = sum(permutations) for nxn matrix.
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        # torch.det: Computes determinant, e.g., [[4, 7], [2, 6]] -> 10.
        # Intent: Calculate det(A) to check invertibility.
        # Purpose: Efficient determinant via LU decomposition.
        return torch.det(A)
