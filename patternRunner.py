import numpy as np
from subprocess import Popen
import os
from plotter import animate_colormap

CWD = os.getcwd()
if os.name == 'nt':
    debugBin = "{}\\x64\\Debug\\TuringPatterns.exe".format(CWD)
    releaseBin = "{}\\x64\\Release\\TuringPatterns.exe".format(CWD)

    GRID_FILE = "{}\\grid.npy".format(CWD)
    INITIAL_CONDITION_FILE = "{}\\ic.npy".format(CWD)

    X_GRID_FILE = "{}\\x_grid.npy".format(CWD)
    Y_GRID_FILE = "{}\\y_grid.npy".format(CWD)
else:
    debugBin = "{}/cmake-build-gcc-debug/TuringPatterns".format(CWD)
    releaseBin = "{}/cmake-build-gcc-release/TuringPatterns".format(CWD)

    GRID_FILE = "{}/grid.npy".format(CWD)
    INITIAL_CONDITION_FILE = "{}/ic.npy".format(CWD)

    X_GRID_FILE = "{}/x_grid.npy".format(CWD)
    Y_GRID_FILE = "{}/y_grid.npy".format(CWD)

chosenBin = releaseBin

def read_solution(file_name, N, N_x, N_y):
    _tensor = np.load(file_name).flatten()
    tensor = []
    for n in range(N):
        m = np.zeros((N_x, N_y))
        for i in range(N_x):
            for j in range(N_y):
                m[i, j] = _tensor[i + j * N_x + n * N_x * N_y]
        tensor.append(m)
    return np.array(tensor)


def run_gray_scott_bacteria(run=True, save=False):
    N = 100
    N_x = 64
    N_y = 64
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "GrayScott"] +
                  ["-bc", "Periodic"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_y)] +
                  ["-nIter", str(N)] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "1.0"] +
                  ["-wns", ".05"] +
                  ["-ud", ".16"] +
                  ["-vd", ".08"] +
                  ["-p1", "0.035"] +
                  ["-p2", ".065"] +
                  ["-of", "bacteria.npy"])
        p.communicate()

    tensor = read_solution("bacteria.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, N_x),
                     np.linspace(0.0, 1.0, N_y),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="bacteria.gif")


def run_gray_scott_bacteria2(run=True, save=False):
    N = 100
    N_x = 64
    N_y = 64
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "GrayScott"] +
                  ["-bc", "Periodic"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_y)] +
                  ["-nIter", str(N)] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "1.0"] +
                  ["-wns", ".05"] +
                  ["-ud", ".14"] +
                  ["-vd", ".06"] +
                  ["-p1", "0.035"] +
                  ["-p2", ".065"] +
                  ["-of", "bacteria2.npy"])
        p.communicate()

    tensor = read_solution("bacteria2.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="bacteria2.gif")


def run_gray_scott_coral(run=True, save=False):
    N = 100
    N_x = 100
    N_y = 100
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "GrayScott"] +
                  ["-bc", "Periodic"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_y)] +
                  ["-nIter", str(N)] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", ".5"] +
                  ["-wns", ".05"] +
                  ["-ud", ".19"] +
                  ["-vd", ".05"] +
                  ["-p1", ".06"] +
                  ["-p2", ".02"] +
                  ["-of", "coral.npy"])
        p.communicate()

    tensor = read_solution("coral.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="coral.gif")


def run_gray_scott_coral2(run=True, save=False):
    N = 100
    N_x = 100
    N_y = 100
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "GrayScott"] +
                  ["-bc", "ZeroFlux"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_y)] +
                  ["-nIter", str(N)] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", ".5"] +
                  ["-wns", ".05"] +
                  ["-ud", ".19"] +
                  ["-vd", ".05"] +
                  ["-p1", ".01"] +
                  ["-p2", ".015"] +
                  ["-of", "coral2.npy"])
        p.communicate()

    tensor = read_solution("coral2.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="coral2.gif")


def run_gray_scott_coral3(run=True, save=False):
    N = 100
    N_x = 100
    N_y = 100
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "GrayScott"] +
                  ["-bc", "Periodic"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_x)] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", ".5"] +
                  ["-wns", ".05"] +
                  ["-ud", ".19"] +
                  ["-vd", ".05"] +
                  ["-p1", ".03"] +
                  ["-p2", ".025"] +
                  ["-of", "coral3.npy"])
        p.communicate()

    tensor = read_solution("coral3.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="coral3.gif")


def run_gray_scott_lines(run=True, save=False):
    N = 100
    N_x = 100
    N_y = 100
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "GrayScott"] +
                  ["-bc", "Periodic"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_x)] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "1"] +
                  ["-wns", ".05"] +
                  ["-ud", ".16"] +
                  ["-vd", ".08"] +
                  ["-p1", ".05"] +
                  ["-p2", ".065"] +
                  ["-of", "lines.npy"])
        p.communicate()

    tensor = read_solution("lines.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="lines.gif")


def run_brussellator_stripes(run=True, save=False):
    N = 100
    N_x = 64
    N_y = 64
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "Brussellator"] +
                  ["-bc", "ZeroFlux"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_y)] +
                  ["-nIter", str(N)] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "0.01"] +
                  ["-wns", ".05"] +
                  ["-ud", "2"] +
                  ["-vd", "16"] +
                  ["-p1", "4.5"] +
                  ["-p2", "7.5"] +
                  ["-of", "br_stripes.npy"])
        p.communicate()

    tensor = read_solution("br_stripes.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="br_stripes.gif")


def run_brussellator_dots(run=True, save=False):
    N = 100
    N_x = 100
    N_y = 100
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "Brussellator"] +
                  ["-bc", "Periodic"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_x)] +
                  ["-nIter", str(N)] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "0.005"] +
                  ["-wns", ".05"] +
                  ["-ud", "2"] +
                  ["-vd", "16"] +
                  ["-p1", "4.5"] +
                  ["-p2", "12"] +
                  ["-of", "br_dots.npy"])
        p.communicate()

    tensor = read_solution("br_dots.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='Spectral',
                     show=not save,
                     save=save,
                     name="br_dots.gif")


def run_schnakenberg(run=True, save=False):
    N = 100
    N_x = 64
    N_y = 64
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "Schnakenberg"] +
                  ["-bc", "ZeroFlux"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_x)] +
                  ["-nIter", str(N)] +
                  ["-nIterPerRound", "200"] +
                  ["-dt", "0.01"] +
                  ["-wns", ".05"] +
                  ["-ud", "1.0"] +
                  ["-vd", "10"] +
                  ["-p1", ".1"] +
                  ["-p2", ".9"] +
                  ["-of", "schnakenberg.npy"])
        p.communicate()

    tensor = read_solution("schnakenberg.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='seismic',
                     show=not save,
                     save=save,
                     name="schnakenberg.gif")


def run_thomas(run=True, save=False):
    N = 100
    N_x = 64
    N_y = 64
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "Thomas"] +
                  ["-bc", "ZeroFlux"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_x)] +
                  ["-nIter", str(N)] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "0.0005"] +
                  ["-wns", ".1"] +
                  ["-ud", "1.0"] +
                  ["-vd", "28"] +
                  ["-p1", "150"] +
                  ["-p2", "100"] +
                  ["-of", "thomas.npy"])
        p.communicate()

    tensor = read_solution("thomas.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='seismic',
                     show=not save,
                     save=save,
                     name="schnakenberg.gif")


def run_fitz_hugh_nagumo(run=True, save=False):
    N = 100
    N_x = 64
    N_y = 64
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "FitzHughNagumo"] +
                  ["-bc", "ZeroFlux"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_x)] +
                  ["-nIter", str(N)] +
                  ["-nIterPerRound", "1000"] +
                  ["-dt", "0.001"] +
                  ["-wns", ".05"] +
                  ["-ud", "1"] +
                  ["-vd", "100"] +
                  ["-p1", "-0.005"] +
                  ["-p2", "10.0"] +
                  ["-of", "fhn.npy"])
        p.communicate()

    tensor = read_solution("fhn.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="fhn.gif")


def run_fitz_hugh_nagumo_low_beta(run=True, save=False):
    N = 100
    N_x = 100
    N_y = 100
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "FitzHughNagumo"] +
                  ["-bc", "ZeroFlux"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_x)] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "1000"] +
                  ["-dt", "0.001"] +
                  ["-wns", ".05"] +
                  ["-ud", "1"] +
                  ["-vd", "100"] +
                  ["-p1", "0.01"] +
                  ["-p2", ".25"] +
                  ["-of", "fhnb.npy"])
        p.communicate()

    tensor = read_solution("fhnb.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="fhnb.gif")


def run_fitz_hugh_nagumo_spatial(run=True, save=False):
    N = 100
    N_x = 100
    N_y = 100
    if run:
        p = Popen([chosenBin] +
                  ["-pattern", "FitzHughNagumo"] +
                  ["-bc", "ZeroFlux"] +
                  ["-xd", str(N_x)] +
                  ["-yd", str(N_x)] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "1000"] +
                  ["-dt", "0.001"] +
                  ["-wns", ".05"] +
                  ["-ud", "1"] +
                  ["-vd", "100"] +
                  ["-p1", "0.01"] +
                  ["-p2", "10"] +
                  ["-of", "fhns.npy"])
        p.communicate()

    tensor = read_solution("fhns.npy", N, N_x, N_y)

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="fhns.gif")


if __name__ == "__main__":
    # run_gray_scott_bacteria(run=True, save=False)
    # run_gray_scott_bacteria2(run=True, save=False)
    # run_gray_scott_coral(run=True, save=False)
    # run_gray_scott_coral2(run=True, save=False)
    # run_gray_scott_coral3(run=False, save=False)
    # run_gray_scott_lines(run=True, save=False)
    # run_brussellator_stripes(run=True, save=False)
    # run_brussellator_dots(run=True, save=False)
    # run_schnakenberg(run=True, save=False)
    # run_thomas(run=True, save=False)
    run_fitz_hugh_nagumo(run=True, save=False)