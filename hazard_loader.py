from PIL import Image
import numpy as np

def load_hazards(path):
    img = Image.open(path).convert("RGB")
    data = np.array(img)

    hazards = {}

    for i in range(len(data)):
        for j in range(len(data[0])):
            r, g, b = data[i][j]

            # Example color detection (you adjust based on your image)
            if r > 200 and g < 50 and b < 50:
                hazards[(i, j)] = "pit"

            elif g > 200 and r < 50:
                hazards[(i, j)] = "teleport"

            elif b > 200:
                hazards[(i, j)] = "confusion"

    return hazards
