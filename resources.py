import torch

DATA_PATH = "./data/WUMI2024a_wildfires_1984_2024_with_subfires.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")