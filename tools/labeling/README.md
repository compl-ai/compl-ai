# Labeling Inspector UI

## Purpose
This directory will house a User Interface (UI) designed to inspect, visualize, and analyze the properties and metadata of the labeled datasets used in the `compl-ai` repository. 

Rather than relying on automated scripts to blindly process labeled data, this tool provides visual insights (e.g., heatmaps, plots, tag distributions) to ensure the integrity of the evaluation pipeline.

## Primary Goals
1. **Balance & Coverage Analysis (Iterative Improvement):**
   - Visualize the distribution of taxonomy tags across the dataset.
   - Identify categories or domains that are underrepresented so that additional benchmarks or datasets can be sourced to fill the gaps.
   
2. **Quality Assurance (Human Review):**
   - Provide an interface for human reviewers to spot-check the labeled data.
   - Verify that annotators applied the schemas correctly and maintained high labeling quality.

## Future Implementation
*Implementation is currently on hold.* 

This directory will eventually be populated with:
1. The formal metadata schema and labeling instructions.
2. The UI application code (e.g., a dashboard for exploring the JSONL outputs).
