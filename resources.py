import torch
import os

DATA_PATH = "./data/WUMI2024a_wildfires_1984_2024_with_subfires.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = "./outputs"
OUTPUT_TABLES_PATH = os.path.join(OUTPUT_PATH, "tables")
OUTPUT_FIGURES_PATH = os.path.join(OUTPUT_PATH, "figures")
MODELS_PATH = os.path.join(OUTPUT_PATH, "models")