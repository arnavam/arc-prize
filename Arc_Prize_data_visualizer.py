import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time

from helper_arc import norm, cmap

# --- Configuration and Styling ---
st.set_page_config(
    page_title="ARC Dataset Viewer with RL Predictions",
    layout="wide",
    initial_sidebar_state="expanded",
)

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

# --- Helper Functions ---
def plot_grid(grid):

    grid = np.array(grid)

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=cmap,norm=norm)

    # Add grid lines between cells
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])


    plt.axis("off")
    return fig

def visualize_prediction_grid(grid_data):
    """Creates a matplotlib figure to display a single grid state for predictions."""
    if not grid_data or not isinstance(grid_data[0], list):
        # Handle empty or malformed grid data
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
        return fig

    grid = np.array(grid_data)
    

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap=cmap, norm=norm)

    # Add grid lines for clarity
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    return fig

@st.cache_data
def load_prediction_data(file_path):
    """Loads the prediction data from the JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the same directory as the script.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None

def main(
    CHALLENGES_FILE='arc-prize-2025/arc-agi_training_challenges.json',
    SOLUTIONS_FILE='arc-prize-2025/arc-agi_training_solutions.json',
    PREDICTIONS_FILE='output-2.json'
):
    st.title("ARC Dataset & Predictions Viewer")

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

    # Load prediction data
    prediction_data = load_prediction_data(PREDICTIONS_FILE)

    # Get all task IDs
    task_ids = list(challenges_data.keys())
    
    # Select task
    selected_id = st.sidebar.selectbox("Select Task ID", task_ids)
    challenge_task = challenges_data[selected_id]
    solution_task = solutions_data.get(selected_id, {})
    
    # Select section: train or test
    section = st.sidebar.radio("Select Section", ["train", "test"])
    
    # Select example
    example_idx = st.sidebar.selectbox("Select Example", range(len(challenge_task[section])), format_func=lambda x: f"Example {x+1}")
    
    # Get the selected example
    pair = challenge_task[section][example_idx]
    
    # Get output for the selected example
    if section == "train":
        output = pair["output"]
    else:  # test section
        try:
            output = solution_task["test"][example_idx]["output"]
        except (KeyError, IndexError, TypeError):
            st.warning("No solution found for this test example.")
            output = None
    
    # Create three columns for side-by-side comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Input")
        fig = plot_grid(pair["input"])
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.header("Target Output")
        if output is not None:
            fig = plot_grid(output)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No target output available")
    
    with col3:
        st.header("RL Agent Predictions")
        
        if section!= "train":
            st.info("Predictions are only available for training examples.")
        elif prediction_data is None:
            
            st.warning("No prediction data available. Please check if the prediction file exists.")
        elif selected_id not in prediction_data:
            st.warning(f"No prediction data available for task {selected_id}.")
        else:
            # Get episodes for this task
            episode_ids = list(prediction_data[selected_id].keys())
            
            if not episode_ids:
                st.warning("No episodes found for this task.")
            else:
                # Find the episode that corresponds to the selected example
                target_episode = str(example_idx)
                
                if target_episode not in episode_ids:
                    st.warning(f"No prediction found for example {example_idx+1} (episode {target_episode}).")
                    # Show available episodes
                    st.info(f"Available episodes: {', '.join(episode_ids)}")
                else:
                    sequence = prediction_data[selected_id][target_episode]
                    num_steps = len(sequence)
                    
                    
                    grid_placeholder = st.empty()
                    score_placeholder = st.empty()
                    vis_mode = st.radio("", ("🎬 Animate Sequence", "👣 Step-by-Step"), horizontal=True)
                    st.info(f"Episode **{target_episode}** (Example {example_idx+1}) has **{num_steps}** steps.")

                    if vis_mode == "🎬 Animate Sequence":
                        
                        # Animation speed control
                        animation_speed = st.slider("Animation Speed (delay in seconds)", 0.1, 2.0, 0.5, 0.1)
                        
                        if st.button("▶️ Play Animation", use_container_width=True):
                            # Placeholders to update dynamically
                            image_placeholder = st.empty()
                            progress_bar = st.progress(0)
                            cols = st.columns(2)
                            step_indicator = cols[0].empty()
                            score_indicator = cols[1].empty()
                            
                            for i, step_data in enumerate(sequence):
                                grid_state = step_data[0]
                                similarity_score = step_data[1]
                                
                                # Update indicators
                                progress_percentage = (i + 1) / num_steps
                                progress_bar.progress(progress_percentage)
                                step_indicator.markdown(f"**Step:** {i + 1}/{num_steps}")
                                score_indicator.metric(label="Similarity Score", value=f"{similarity_score:.4f}")
                                
                                # Update grid image
                                fig = visualize_prediction_grid(grid_state)
                                grid_placeholder.pyplot(fig)
                                plt.close(fig)
                                
                                time.sleep(animation_speed)
                            
                            st.success("Animation complete! 🎉")
                    
                    elif vis_mode == "👣 Step-by-Step":
                        
                        # Slider to select a specific step
                        step_index = st.slider("Select a step to display", 0, num_steps - 1, num_steps - 1)
                        
                        grid_state = sequence[step_index][0]
                        similarity_score = sequence[step_index][1]
                        
                        # Display the metric and the grid
                        fig = visualize_prediction_grid(grid_state)
                        grid_placeholder.pyplot(fig)
                        plt.close(fig)

                        st.metric(label=f"Similarity Score (at step {step_index + 1})", value=f"{similarity_score:.4f}")

                        


if __name__ == "__main__":
    main()