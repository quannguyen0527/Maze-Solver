from PIL import Image
import numpy as np

def load_hazards(path):
    img = Image.open(path).convert("RGB")
    data = np.array(img)

    hazards = {}

    for i in range(len(data)):
        for j in range(len(data[0])):
            r, g, b = data[i][j]

            # STRICT detection (NO ranges)
            if (r, g, b) == (255, 0, 0):
                hazards[(i, j)] = "pit"

            elif (r, g, b) == (0, 255, 0):
                hazards[(i, j)] = "teleport"

            elif (r, g, b) == (0, 0, 255):
                hazards[(i, j)] = "confusion"

    return hazards