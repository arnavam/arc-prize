import numpy
from helper_arc import loader
import json
import os

# Load dataset
train, ids = loader(dataset_path='arc-prize-2025/arc-agi_training_challenges.json')

def matrix_to_string(matrix): #function to print a matrix as string
    return "\n".join(" ".join(str(cell) for cell in row) for row in matrix)

def format_task(task, task_id, task_number): #function to format a task into string
    output = []
    output.append(f"\n=== Task {task_number}: ID = {task_id} ===")

    output.append("\n-- Train Examples --")
    for i, pair in enumerate(task['train']):
        output.append(f"\nTrain Example {i}:")
        output.append("Input:")
        output.append(matrix_to_string(pair['input']))
        output.append("Output:")
        output.append(matrix_to_string(pair['output']))

    output.append("\n-- Test Examples --")
    for i, pair in enumerate(task['test']):
        output.append(f"\nTest Example {i}:")
        output.append("Input:")
        output.append(matrix_to_string(pair['input']))
        if 'output' in pair:
            output.append("Output:")
            output.append(matrix_to_string(pair['output']))

    return "\n".join(output)


# Create output file
output_filename = "arc_tasks_visualization.txt"
with open(output_filename, "w") as f:
    for idx, task_id in enumerate(ids):
        task = train[task_id]
        task_str = format_task(task, task_id, idx + 1)
        f.write(task_str)
        f.write("\n" + "=" * 40 + "\n")  # separator between tasks

print(f"All task outputs written to '{output_filename}'")
