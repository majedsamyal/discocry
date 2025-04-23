| **Aspect**            | **Systems of Linear Equations** | **Linear Regression** | **Neural Networks** |
|-----------------------|---------------------------------|-----------------------|---------------------|
| **Equation Form**     | \( Ax = b \)                   | \( X w = y \), solved as \( X^T X w = X^T y \) | Forward: \( z = W x + b \), Training: \( X w = y \) (simplified) |
| **Matrix Structure**  | \( A \): Coefficient matrix, \( x \): Variables, \( b \): Constants | \( X \): Feature matrix, \( w \): Weights, \( y \): Targets | \( W \): Weight matrix, \( x \): Input, \( b \): Bias, \( z \): Output |
| **Purpose**           | Find exact solution for variables | Find best-fit weights for prediction | Compute layer outputs, learn weights for prediction |
| **Solution Method**   | Direct (e.g., Gaussian elimination, matrix inverse) | Normal equation or iterative (e.g., gradient descent) | Iterative (e.g., backpropagation, gradient descent) |
| **System Type**       | Square (if solvable), over/underdetermined possible | Typically overdetermined, solved approximately | Overdetermined in training, iterative optimization |
| **Use in AI/ML**      | Foundation for solving linear systems | Predict continuous outputs (e.g., house prices) | Model complex patterns (e.g., image classification) |
| **Example**           | \( 2x + y = 5, x + 3y = 4 \)  | Fit \( y = wx \) to \( (1, 2), (2, 4) \) | Layer: \( z = w_1 x_1 + w_2 x_2 \) for \( (1, 1, 3), (2, 0, 4) \) |

