from PIL import Image
import numpy as np

class MazeLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        img = Image.open(self.path).convert("RGB")
        data = np.array(img)

        grid = []
        for row in data:
            grid_row = []
            for (r, g, b) in row:
                # Wall = dark pixel
                if r < 50 and g < 50 and b < 50:
                    grid_row.append(1)
                else:
                    grid_row.append(0)
            grid.append(grid_row)

        return np.array(grid)

    def print_sample(self, grid, size=20):
        for i in range(size):
            print("".join(str(cell) for cell in grid[i][:size]))
