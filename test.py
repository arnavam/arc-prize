with col3:
    st.header("RL Agent Predictions")
    
    if section != "train":
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
                
                st.info(f"Episode **{target_episode}** (Example {example_idx+1}) has **{num_steps}** steps.")
                
                # Show the current prediction grid at the top
                grid_placeholder = st.empty()
                score_placeholder = st.empty()
                
                # Option to play as animation or view step-by-step
                vis_mode = st.radio("Visualization Mode:", ("🎬 Animate Sequence", "👣 Step-by-Step"), horizontal=True)
                
                if vis_mode == "🎬 Animate Sequence":
                    st.markdown("Click the button below to play the sequence of predictions.")
                    
                    # Animation speed control
                    animation_speed = st.slider("Animation Speed (delay in seconds)", 0.1, 2.0, 0.5, 0.1)
                    
                    if st.button("▶️ Play Animation", use_container_width=True):
                        # Placeholders to update dynamically
                        progress_bar = st.progress(0)
                        step_cols = st.columns(2)
                        step_indicator = step_cols[0].empty()
                        
                        for i, step_data in enumerate(sequence):
                            grid_state = step_data[0]
                            similarity_score = step_data[1]
                            
                            # Update indicators
                            progress_percentage = (i + 1) / num_steps
                            progress_bar.progress(progress_percentage)
                            step_indicator.markdown(f"**Step:** {i + 1}/{num_steps}")
                            score_placeholder.metric(label="Similarity Score", value=f"{similarity_score:.4f}")
                            
                            # Update grid image
                            fig = visualize_prediction_grid(grid_state)
                            grid_placeholder.pyplot(fig)
                            plt.close(fig)
                            
                            time.sleep(animation_speed)
                        
                        st.success("Animation complete! 🎉")
                
                elif vis_mode == "👣 Step-by-Step":
                    st.markdown("Use the slider to manually inspect each step of the prediction sequence.")
                    
                    # Slider to select a specific step
                    step_index = st.slider("Select a step to display", 0, num_steps - 1, num_steps - 1)
                    
                    grid_state = sequence[step_index][0]
                    similarity_score = sequence[step_index][1]
                    
                    # Display the grid and score in the placeholders at the top
                    fig = visualize_prediction_grid(grid_state)
                    grid_placeholder.pyplot(fig)
                    plt.close(fig)
                    score_placeholder.metric(label=f"Similarity Score (at step {step_index + 1})", value=f"{similarity_score:.4f}")