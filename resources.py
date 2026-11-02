import torch
import os

DATA_PATH = "./data/WUMI2024a_wildfires_1984_2024_with_subfires.txt"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
OUTPUT_PATH = "./outputs"
OUTPUT_TABLES_PATH = os.path.join(OUTPUT_PATH, "tables")
OUTPUT_FIGURES_PATH = os.path.join(OUTPUT_PATH, "figures")
MODELS_PATH = os.path.join(OUTPUT_PATH, "models")