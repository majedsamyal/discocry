import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        # input_dim = 3 (sugar, salt, spice)
        # output_dim = 2 (sweetness_score, savory_score)
 
        # This is to initialise Weight and bias
        # We need to define W and b required for equation : y = W * x + b 
        # x will be input feature matrix
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros((output_dim, 1)) # Always 1 column
        
    def forward(self, x):
        # x = ingredient amounts for multiple recipes
        # e.g., [[5, 3, 8],    <- Recipe 1: sugar=5, salt=2, spice=4
        #        [2, 1, 3],    <- Recipe 2: sugar=3, salt=1, spice=3
        #        [4, 2, 1]]    <- Recipe 3: sugar=8, salt=2, spice=1
       
        # This is feature matrix which we will transfer with weight
        self.input = x  
        
        # Calculate taste scores for each recipe:
        # For each recipe: sweetness = (sugar×w11 + salt×w12 + spice×w13) + bias
        #                  savory = (sugar×w21 + salt×w22 + spice×w23) + bias
        # Based on weigths we will get the final score as two outputs
        # output = W * x + b 
        self.output = np.dot(self.W, x) + self.b
        return self.output
    
    def backward(self, grad_output, learning_rate=0.01):
        # grad_output = how wrong our predictions were
        # e.g., [[2, 1, -1],   <- Sweetness was: 2 too high, 1 too high, 1 too low
        #        [-1, 0, 3]]   <- Savory was: 1 too low, perfect, 3 too high
        
        # How many recipes did we test?
        batch_size = self.input.shape[1]
        
        # FINDING WHO TO BLAME (grad_W):
        # "Sweetness was 2 points too high on recipe 1, and recipe 1 had 5 sugar"
        # So sugar's contribution to sweetness should decrease proportionally
        # This calculates that blame for EVERY weight
        self.grad_W = np.dot(grad_output, self.input.T) / batch_size
        
        # FIXING THE BASELINE (grad_b):
        # If sweetness is consistently too high across all recipes,
        # lower the base sweetness score
        self.grad_b = np.sum(grad_output, axis=1, keepdims=True) / batch_size
        
        # TELL THE KITCHEN (grad_input):
        # "Hey ingredients! Here's how much each of you contributed to the error"
        # This lets earlier layers (if any) know what to fix
        grad_input = np.dot(self.W.T, grad_output)
        
        # ADJUST THE RECIPE RULES:
        # Slightly adjust how much we think each ingredient affects each taste
        # learning_rate = 0.01 means "only move 1% in that direction" (don't overcorrect)
        self.W -= learning_rate * self.grad_W  # Fix the ingredient→taste rules
        self.b -= learning_rate * self.grad_b  # Fix the baseline scores
        
        # Pass the blame back to whoever gave us these ingredients
        return grad_input

    # Create our recipe predictor
# 3 ingredients → 2 taste scores
# layer = LinearLayer(3, 2)

# Test 4 recipes at once
# Each column is a recipe: [sugar, salt, spice]
# recipes = np.array([[5, 3, 8, 2],   # sugar amounts
#                     [2, 1, 2, 4],   # salt amounts  
#                     [4, 3, 1, 6]])  # spice amounts
#
# Predict taste scores
# predictions = layer.forward(recipes)
# Output: [[sweetness for each recipe],
#          [savory for each recipe]]

# We taste test and find our predictions were wrong:
# (positive = predicted too high, negative = predicted too low)
# errors = np.array([[2, 1, -1, 0],   # sweetness errors
#                    [-1, 0, 3, 2]])   # savory errors
#
# Learn from mistakes
# layer.backward(errors)
# Now weights are slightly adjusted - sugar's effect on sweetness decreased
# because it contributed to overestimating sweetness in recipes 1 and 2
