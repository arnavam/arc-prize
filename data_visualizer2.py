import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

# --- Configuration and Styling ---
st.set_page_config(
    page_title="RL Agent Prediction Visualizer",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🤖 RL Agent Prediction Visualizer")
st.markdown("Visualize the sequential state predictions from your Reinforcement Learning model.")

# --- Helper Functions ---

@st.cache_data
def load_data(file_path):
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

def visualize_grid(grid_data):
    """Creates a matplotlib figure to display a single grid state."""
    if not grid_data or not isinstance(grid_data[0], list):
        # Handle empty or malformed grid data
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
        return fig

    grid = np.array(grid_data)
    
    # Create a discrete colormap (0=black, 1=blue, 2=red, etc.)
    colors = ['black', 'blue', 'red', 'green', 'yellow', 'grey', 'magenta', 'orange', 'cyan', 'brown']
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, 10.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap=cmap, norm=norm)

    # Add grid lines for clarity
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    return fig

# --- Main Application Logic ---

# Load the data
prediction_data = load_data('output.json')

if prediction_data:
    # --- Sidebar for Navigation ---
    st.sidebar.header("Navigation Controls")
    task_ids = list(prediction_data.keys())
    selected_task = st.sidebar.selectbox("1. Select a Task ID", task_ids)

    if selected_task:
        episode_ids = list(prediction_data[selected_task].keys())
        selected_episode = st.sidebar.selectbox("2. Select an Episode ID", episode_ids)

        if selected_episode:
            sequence = prediction_data[selected_task][selected_episode]
            num_steps = len(sequence)

            st.sidebar.info(f"Task **{selected_task}** | Episode **{selected_episode}** selected.\n\nThis episode has **{num_steps}** steps.")

            # --- Main Content Area for Visualization ---
            st.header(f"Visualizing Episode: `{selected_episode}`")

            # Option to play as animation or view step-by-step
            vis_mode = st.radio("Choose Visualization Mode:", ("🎬 Animate Sequence", "👣 Step-by-Step"), horizontal=True)

            if vis_mode == "🎬 Animate Sequence":
                st.markdown("Click the button below to play the sequence of predictions as a video.")
                
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
                        fig = visualize_grid(grid_state)
                        image_placeholder.pyplot(fig)
                        plt.close(fig)  # Important to close plot to free memory

                        time.sleep(animation_speed)
                    
                    st.success("Animation complete! 🎉")

            elif vis_mode == "👣 Step-by-Step":
                st.markdown("Use the slider to manually inspect each step of the prediction sequence.")
                
                # Slider to select a specific step
                step_index = st.slider("Select a step to display", 0, num_steps - 1, 0)
                
                grid_state = sequence[step_index][0]
                similarity_score = sequence[step_index][1]

                # Display the metric and the grid
                st.metric(label=f"Similarity Score (at step {step_index + 1})", value=f"{similarity_score:.4f}")
                fig = visualize_grid(grid_state)
                st.pyplot(fig)
                plt.close(fig)