import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
from sklearn.cluster import KMeans
from itertools import product

# --- Neural Feature Extractor ---
def create_feature_extractor(input_shape=(30, 30, 1)):
    """Creates a CNN model for feature extraction from grids"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, name='feature_vector')
    ])
    return model

# --- Symbolic Program Generator ---
PRIMITIVES = [
    'rotate', 'mirrorlr', 'mirrorud', 'lcrop', 'rcrop', 'ucrop', 'dcrop',
    'recolor', 'select', 'fill', 'overlay', 'resize'
]

def generate_programs(max_length=3):
    """Generates symbolic programs of primitive operations"""
    programs = []
    for length in range(1, max_length + 1):
        for combo in product(PRIMITIVES, repeat=length):
            programs.append(list(combo))
    return programs

# --- Differentiable Program Executor ---
class ProgramExecutor(tf.keras.Model):
    """Differentiable executor for symbolic programs"""
    def __init__(self, feature_extractor, num_primitives):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.primitive_weights = layers.Dense(num_primitives, activation='softmax')
        
    def call(self, inputs):
        # Extract features from input/output pairs
        input_grid, output_grid = inputs
        input_feat = self.feature_extractor(input_grid)
        output_feat = self.feature_extractor(output_grid)
        
        # Compute primitive probabilities
        combined = tf.concat([input_feat, output_feat], axis=-1)
        return self.primitive_weights(combined)

# --- Neural-Symbolic Solver ---
class NeuralSymbolicSolver:
    def __init__(self):
        self.feature_extractor = create_feature_extractor()
        self.executor = ProgramExecutor(self.feature_extractor, len(PRIMITIVES))
        self.executor.compile(
            optimizer='adam',
            loss='categorical_crossentropy'
        )
        self.programs = generate_programs(max_length=3)
        
    def train(self, train_data, epochs=10, batch_size=32):
        """Train on demonstration pairs"""
        # Prepare training data
        X_in, X_out, y_primitive = [], [], []
        
        for example in train_data:
            input_grid = self.preprocess(example['input'])
            output_grid = self.preprocess(example['output'])
            
            # Find best primitive (simplified for example)
            best_primitive = self.find_best_primitive(input_grid, output_grid)
            
            X_in.append(input_grid)
            X_out.append(output_grid)
            y_primitive.append(PRIMITIVES.index(best_primitive))
        
        # Train the model
        X_in = np.array(X_in)
        X_out = np.array(X_out)
        y_primitive = tf.keras.utils.to_categorical(y_primitive, len(PRIMITIVES))
        
        self.executor.fit(
            [X_in, X_out], y_primitive,
            epochs=epochs,
            batch_size=batch_size
        )
    
    def solve(self, input_grid, output_grid):
        """Solve a new problem using neural-guided program synthesis"""
        # Predict primitive probabilities
        input_pp = self.preprocess(input_grid)
        output_pp = self.preprocess(output_grid)
        primitive_probs = self.executor.predict(
            [np.array([input_pp]), np.array([output_pp])]
        )[0]
        
        # Rank programs by primitive probabilities
        program_scores = []
        for program in self.programs:
            score = np.prod([primitive_probs[PRIMITIVES.index(p)] for p in program])
            program_scores.append((program, score))
        
        # Try top programs
        program_scores.sort(key=lambda x: x[1], reverse=True)
        
        for program, _ in program_scores[:10]:  # Try top 10
            result = self.execute_program(input_grid, program)
            if np.array_equal(result, output_grid):
                return program
        
        return None  # No solution found
    
    def execute_program(self, grid, program):
        """Execute a symbolic program on a grid"""
        current = grid.copy()
        for op in program:
            if op == 'rotate':
                current = np.rot90(current, k=-1)
            elif op == 'mirrorlr':
                current = np.fliplr(current)
            elif op == 'mirrorud':
                current = np.flipud(current)
            elif op == 'lcrop':
                current = current[:, 1:] if current.shape[1] > 1 else current
            elif op == 'rcrop':
                current = current[:, :-1] if current.shape[1] > 1 else current
            elif op == 'ucrop':
                current = current[1:, :] if current.shape[0] > 1 else current
            elif op == 'dcrop':
                current = current[:-1, :] if current.shape[0] > 1 else current
            elif op == 'recolor':
                current = self.learn_recoloring(current)
            # Additional operations would be implemented here
        return current
    
    def learn_recoloring(self, grid):
        """Learn color mapping using clustering (simplified)"""
        # In practice, this would compare input/output colors
        return grid  # Placeholder
    
    def find_best_primitive(self, input_grid, output_grid):
        """Find best matching primitive (simplified heuristic)"""
        # In practice, use neural features for this
        for primitive in ['rotate', 'mirrorlr', 'mirrorud']:
            transformed = self.execute_program(input_grid, [primitive])
            if np.array_equal(transformed, output_grid):
                return primitive
        return 'recolor'  # Default
    
    def preprocess(self, grid, size=30):
        """Preprocess grid to fixed size with padding"""
        h, w = len(grid), len(grid[0])
        padded = np.zeros((size, size), dtype=int)
        padded[:h, :w] = grid
        return np.expand_dims(padded, axis=-1)

# --- Main Execution ---
if __name__ == "__main__":
    # Load training data
    with open('arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
        train_data = json.load(f)
    
    # Prepare training examples
    training_examples = []
    for case_id, case_data in train_data.items():
        for example in case_data['train']:
            training_examples.append({
                'input': example['input'],
                'output': example['output']
            })
    
    # Initialize and train solver
    solver = NeuralSymbolicSolver()
    solver.train(training_examples[:100], epochs=5, batch_size=16)
    
    # Test on a sample case
    sample_case = list(train_data.values())[0]
    input_grid = sample_case['train'][0]['input']
    output_grid = sample_case['train'][0]['output']
    
    solution = solver.solve(input_grid, output_grid)
    print(f"Solution: {solution}")
    
    # Visualize results
    if solution:
        result = solver.execute_program(input_grid, solution)
        print("Original Input:")
        print(np.array(input_grid))
        print("\nSolved Output:")
        print(result)
        print("\nTarget Output:")
        print(np.array(output_grid))