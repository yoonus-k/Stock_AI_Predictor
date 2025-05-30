# Cloud Training Guide for RL Trading Model

This guide provides instructions for training your reinforcement learning trading model on cloud platforms like Kaggle and Hugging Face.

## Why Use Cloud Platforms?

Reinforcement learning is computationally intensive and typically requires GPU acceleration for efficient training. Cloud platforms offer:

- Free GPU access
- Higher computational capacity than most personal laptops
- Persistent storage for large models
- Easy sharing and collaboration features

## Option 1: Kaggle (Recommended for Training)

Kaggle offers 30+ hours of weekly GPU time for free, making it an excellent choice for training complex RL models.

### Setting Up on Kaggle

1. **Create a Kaggle account** if you don't have one at [kaggle.com](https://www.kaggle.com/)

2. **Create a new notebook**:
   - Click "Create" → "Notebook"
   - Set the notebook to use GPU: Click "Settings" on the right sidebar → "Accelerator" → select "GPU"

3. **Upload your dataset**:
   - Go to "Data" tab in your Kaggle account
   - Click "New Dataset"
   - Upload your `samples.db` file and any other necessary data
   - Make note of the path to your dataset (it will be under `/kaggle/input/your-dataset-name/`)

4. **Set up the environment**:
   - Use our setup script by running this in a notebook cell:
   
   ```python
   # Download setup script
   !wget https://raw.githubusercontent.com/yoonus/Stock_AI_Predictor/main/RL/Scripts/kaggle_setup.py
   
   # Run setup script
   %run kaggle_setup.py
   ```

5. **Launch training**:
   ```python
   # Import training function
   from RL.Scripts.kaggle_training import train_model
   
   # Train model with our dataset
   model = train_model(
       db_path='/kaggle/input/your-dataset-name/samples.db',
       output_dir='/kaggle/working/output',
       timesteps=500000  # Increase for better performance
   )
   ```

6. **Download trained model**:
   - After training completes, you'll find the model at `/kaggle/working/output/trading_model.zip`
   - Click on the file in the "Output" tab and download it

## Option 2: Google Colab

If you prefer Google Colab's interface, here's how to use it:

1. **Create a new Colab notebook**:
   - Go to [colab.research.google.com](https://colab.research.google.com/)
   - Create a new notebook
   - Enable GPU: "Runtime" → "Change runtime type" → "Hardware accelerator" → "GPU"

2. **Install dependencies and clone repository**:
   ```python
   # Install stable-baselines3 and other dependencies
   !pip install stable-baselines3 gymnasium pandas matplotlib numpy tqdm scikit-learn shap
   
   # Clone repository
   !git clone https://github.com/yoonus/Stock_AI_Predictor.git
   %cd Stock_AI_Predictor
   ```

3. **Upload your dataset**:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload your samples.db file
   
   # Move to appropriate location
   !mkdir -p RL/Data
   !mv samples.db RL/Data/
   ```

4. **Train model**:
   ```python
   # Add project root to path
   import sys
   sys.path.append('/content/Stock_AI_Predictor')
   
   # Import training script
   from RL.Scripts.kaggle_training import train_model
   
   # Train with GPU acceleration
   model = train_model(
       db_path='/content/Stock_AI_Predictor/RL/Data/samples.db',
       output_dir='/content/output',
       timesteps=200000  # Adjust based on available time
   )
   ```

5. **Save model to Google Drive** (to prevent losing it when session ends):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Copy model to Google Drive
   !cp /content/output/trading_model.zip /content/drive/My\ Drive/
   ```

## Option 3: Hugging Face Spaces (For Deployment)

Hugging Face is excellent for deploying and sharing your trained models.

### Setting Up on Hugging Face

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co/)

2. **Create a new Space**:
   - Go to your profile → "Spaces" → "Create a new Space"
   - Select "Gradio" as the SDK
   - Choose a template that fits your needs
   - Set Resource to "CPU" (or "GPU" if available for your account level)

3. **Upload your trained model**:
   - After training on Kaggle or Colab, upload your model to the Hugging Face Space
   - Use the Files tab or Git to add your model

4. **Create an app for model interaction**:
   - We've provided a basic Gradio app template in `Deployment/app.py`
   - Modify it to load your trained model and handle user inputs

5. **Deploy and share**:
   - Once your app is working, your model will be available via a public URL
   - Share this URL with others to let them use your trading model

## Using GPU Effectively

To ensure you're using the GPU effectively during training:

1. **Use an appropriate batch size** (128-512 for RL models on GPU)
2. **Monitor GPU utilization** using `nvidia-smi` command or platform tools
3. **Set n_steps to a higher value** (2048 or 4096) to process more data per update
4. **Increase model complexity** if GPU utilization is low (deeper networks)

## Saving and Loading Models

To save your trained model and use it elsewhere:

```python
# On Kaggle/Colab (during training)
model.save("/path/to/model.zip")

# On your local machine (after downloading)
from stable_baselines3 import PPO
from RL.Envs.trading_env import PatternSentimentEnv
from RL.Envs.action_wrapper import TupleActionWrapper

# Create environment
env = TupleActionWrapper(PatternSentimentEnv(...))

# Load model
model = PPO.load("path/to/downloaded/model.zip", env=env)

# Use model for predictions
obs = env.reset()[0]
action, _ = model.predict(obs)
```

## Troubleshooting

- **Session disconnects**: Kaggle/Colab may disconnect after periods of inactivity. Use `KeepAlive` solutions or decrease number of timesteps.
- **Out of memory errors**: Reduce batch size or model complexity.
- **Training too slow**: Ensure GPU is being utilized properly; check with monitoring tools.

## Additional Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Kaggle GPU Kernels Guide](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
