import json
from preparation import validate

with open("./config.json", "r") as f:
    config = json.load(f)

if __name__ == "__main__":
    validate()                 # Check the PSNR and SSIM results for SIDD Validation
