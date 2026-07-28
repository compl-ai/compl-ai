# Labeling Inspector UI

A Next.js dashboard for visually inspecting and human-reviewing the LLM-generated labels for the `compl-ai` datasets.

## Features
- **Data Joining**: Dynamically merges raw data from `../datasets/` with the annotations in `../labels/` on the fly.
- **Human Review**: Reviewers can override the LLM's primary/secondary labels and add a rationale.
- **Sparse Approval**: Reviewers can mark specific samples as "Approved" to lock in the annotations and remove them from the "Review Needed" queue.
- **Patch Management**: All human modifications are saved cleanly to `../labels/[dataset]_patch.jsonl` and deduplicated by `sample_id`.

## Getting Started

1. **Ensure Datasets Exist**: You must have generated the raw datasets locally first.
   ```bash
   cd ../datasets
   ./regenerate_datasets.sh
   ```

2. **Install Dependencies**:
   ```bash
   cd ../ui  # (or from within tools/labeling/ui)
   npm install
   ```

3. **Run the Development Server**:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) with your browser to start reviewing.
