# src/practice_exercises.py
import torch
from matrix_operations import MatrixOperations

def run_exercises():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ops = MatrixOperations()
    
    print("=== Discocry: Linear Algebra Practice (Chapter 2) ===")
    
    # Exercise 1: System of Linear Equations (Section 2.1)
    print("\nExercise 1: Solving a System of Linear Equations")
    A = ops.create_matrix(2, 2, [[2, 1], [1, 3]], device=device)
    b = torch.tensor([5, 4], dtype=torch.float32, device=device)
    try:
        x = ops.solve_linear_system(A, b)
        print("System: Ax = b\nA:\n", A, "\nb:\n", b)
        print("Solution x:", x.tolist())
    except RuntimeError:
        print("No unique solution exists.")

    # Exercise 2: Matrix Operations (Section 2.2)
    print("\nExercise 2: Matrix Addition and Multiplication")
    A = ops.create_matrix(2, 2, [[1, 2], [3, 4]], device=device)
    B = ops.create_matrix(2, 2, [[5, 6], [7, 8]], device=device)
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("A + B:\n", ops.add_matrices(A, B))
    print("A * B:\n", ops.matrix_multiply(A, B))

    # Exercise 3: Transpose (Section 2.2)
    print("\nExercise 3: Matrix Transpose")
    A = ops.create_matrix(2, 3, [[1, 2, 3], [4, 5, 6]], device=device)
    print("Matrix A:\n", A)
    print("Transpose A:\n", ops.transpose(A))

    # Exercise 4: Determinant (Section 4.1, referenced in 2.3)
    print("\nExercise 4: Matrix Determinant")
    A = ops.create_matrix(2, 2, [[4, 7], [2, 6]], device=device)
    det = ops.determinant(A)
    print("Matrix A:\n", A)
    print("Determinant:", det.item())

if __name__ == "__main__":
    run_exercises()
