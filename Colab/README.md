# Running Stock_AI_Predictor in Google Colab

This directory contains scripts and notebooks configured to run the Stock_AI_Predictor models in Google Colab.

## Files

- **run_on_colab.ipynb**: The main notebook to run in Google Colab
- **colab_db_helper.py**: Helper module for database connection and setup
- **colab_db_patcher.py**: Utility to patch database paths for Colab compatibility
- **parameter_tester.py**: Standard parameter testing script
- **parameter_tester_multithreaded.py**: Multi-threaded parameter testing for better performance
- **pip_pattern_miner.py**: Pattern mining implementation
- **requirements.txt**: Dependencies required for the Colab environment

## Setup Instructions

1. Upload your project to Google Drive or clone it directly in Colab
2. Make sure your database file (`data.db`) is available in your Google Drive
3. Open `run_on_colab.ipynb` in Google Colab
4. Follow the step-by-step instructions in the notebook

## Database Setup

The notebook will:
1. Mount your Google Drive
2. Copy the database file from Drive to the local Colab environment
3. Patch database connection paths to ensure compatibility
4. Set up the correct directory structure

## Running Models

The notebook provides cells to run:
1. Multi-threaded parameter testing
2. PIP (Perceptually Important Points) pattern mining
3. Standard parameter testing

## Results

After running the models, the notebook will:
1. Create a timestamped directory in your Google Drive
2. Copy all generated results (images, updated database, experiment results)
3. Ensure your work is preserved when the Colab session ends

## Troubleshooting

If you encounter issues:
1. Check that your database path is correct in the notebook
2. Verify that all dependencies are installed correctly
3. Ensure your Google Drive is properly mounted
4. Check for error messages in the notebook output

## Additional Resources

For more information about the Stock_AI_Predictor project, refer to the main README file in the project root directory.
