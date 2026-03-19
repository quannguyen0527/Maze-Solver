import matplotlib.pyplot as plt

def show_maze(grid, path=None):
    plt.imshow(grid, cmap='gray')

    if path:
        x = [p[1] for p in path]
        y = [p[0] for p in path]
        plt.plot(x, y)

    plt.title("Maze Solution")
    plt.show()
