import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt

def render_grid(grid, title):
    grid = np.array(grid)
    cmap = plt.get_cmap("tab20")
    plt.imshow(grid, cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.clf()

def main(
    CHALLENGES_FILE='arc-prize-2025/arc-agi_training_challenges.json',
    SOLUTIONS_FILE='arc-prize-2025/arc-agi_training_solutions.json'
):
    st.title("ARC Prize Dataset Viewer")

    # Load challenge file
    try:
        with open(CHALLENGES_FILE, "r") as f:
            challenges_data = json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {CHALLENGES_FILE}")
        st.stop()

    # Load solution file
    try:
        with open(SOLUTIONS_FILE, "r") as f:
            solutions_data = json.load(f)
    except FileNotFoundError:
        st.warning("Solutions file not found. Test outputs may not be displayed.")
        solutions_data = {}

    # Get all task IDs
    task_ids = list(challenges_data.keys())

    # Select task
    selected_id = st.selectbox("Select Task ID", task_ids)
    challenge_task = challenges_data[selected_id]
    solution_task = solutions_data.get(selected_id, {})

    # Select section: train or test
    section = st.radio("Select Section", ["train", "test"])

    # Iterate through examples
    for i, pair in enumerate(challenge_task[section]):
        st.markdown(f"### {section.capitalize()} Example {i + 1}")
        cols = st.columns(2)

        with cols[0]:
            st.markdown("**Input**")
            render_grid(pair["input"], f"{section.capitalize()} {i+1} Input")

        with cols[1]:
            st.markdown("**Output**")
            if section == "train":
                output = pair["output"]
            else:  # test section
                try:
                    output = solution_task["test"][i]["output"]
                except (KeyError, IndexError, TypeError):
                    st.warning("No solution found for this test example.")
                    output = None

            if output is not None:
                render_grid(output, f"{section.capitalize()} {i+1} Output")

if __name__ == "__main__":
    main()
