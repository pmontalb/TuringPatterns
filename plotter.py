import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from subprocess import Popen
import os


def __ax_formatter(ax, title="", x_label="", y_label="", show_legend=False):
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_legend:
        ax.legend(loc='best')


def surf(z, x, y, title="", x_label="", y_label="", show_legend=False, show=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Create X and Y data
    x_grid, y_grid = np.meshgrid(x, y)

    ax.plot_surface(x_grid, y_grid, z, rstride=1, cstride=1, antialiased=True)

    __ax_formatter(ax, title, x_label, y_label, show_legend)

    if show:
        plt.show()


def plot(z, x, title="", x_label="", y_label="", show_legend=False, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(x)):
        ax.plot(z[:, i])

    __ax_formatter(ax, title, x_label, y_label, show_legend)

    if show:
        plt.show()


def animate(z, x, show=False):
    fig, ax = plt.subplots()

    line, = ax.plot(x, z[:, 0])

    def update_line(i):
        if i >= z.shape[1]:
            return line,
        line.set_ydata(z[:, i])  # update the data
        return line,

    # Init only required for blitting to give a clean slate.
    def init():
        line.set_ydata(z[:, 0])
        return line,

    ani = animation.FuncAnimation(fig, update_line, np.arange(1, 200), init_func=init, interval=25, blit=True)

    if show:
        plt.show()


def read_1D():
    mat = []
    with open(os.getcwd() + "\\results.csv") as f:
        lines = f.readlines()
        for line in lines:
            mat.append([float(x) for x in line.split(",") if x not in ["", "\n"]])
    mat = np.array(mat)
    return mat


if __name__ == "__main__":
    dll = os.getcwd() + "\\x64\\Debug\\TuringPatterns.exe"
    p = Popen(dll)
    p.communicate()

    mat = read_1D()

    #surf(mat, np.arange(mat.shape[1]), np.linspace(0.0, 1.0, mat.shape[0]), show=True)
    #plot(mat, np.arange(mat.shape[1]), show=True)
    animate(mat, np.linspace(0.0, 1.0, mat.shape[0]), show=True)



