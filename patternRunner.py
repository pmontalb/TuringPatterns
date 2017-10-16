import numpy as np
from subprocess import Popen
import os
from plotter import animate_colormap, animate_3D

debugDll = os.getcwd() + "\\x64\\Debug\\TuringPatterns.exe"
releaseDll = os.getcwd() + "\\x64\\Release\\TuringPatterns.exe"


def read_1D(file_name):
    mat = []
    with open(os.getcwd() + "\\" + file_name) as f:
        lines = f.readlines()
        for line in lines:
            mat.append([float(x) for x in line.split(",") if x not in ["", "\n"]])
    mat = np.array(mat)
    return mat


def read_2D(file_name):
    tensor = []

    n_matrices = 0
    n_rows = 0
    with open(os.getcwd() + "\\" + file_name) as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split(",")) == 1:
                n_matrices += 1
                continue
            elif n_matrices == 0:
                n_rows += 1

            tensor.append([float(x) for x in line.split(",") if x not in ["", "\n"]])
    tensor = np.array(tensor)
    tensor = tensor.reshape((n_matrices, n_rows, tensor.shape[1]))
    return tensor


def run_gray_scott_bacteria(run=True):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "GrayScott"] +
                  ["-boundaryCondition", "Periodic"] +
                  ["-xDimension", "64"] +
                  ["-yDimension", "64"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "1.0"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", ".16"] +
                  ["-vDiffusion", ".08"] +
                  ["-patternParameter1", "0.035"] +
                  ["-patternParameter2", ".065"] +
                  ["-solutionFile", "bacteria.csv"])
        p.communicate()

    tensor = read_2D("bacteria.csv")

    animate_colormap(tensor, np.linspace(0.0, 1.0, tensor.shape[2]), np.linspace(0.0, 1.0, tensor.shape[1]), True)


def run_gray_scott_stripes(run=True):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "GrayScott"] +
                  ["-boundaryCondition", "Periodic"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "16"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "1.0"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", ".16"] +
                  ["-vDiffusion", ".08"] +
                  ["-patternParameter1", "0.035"] +
                  ["-patternParameter2", ".065"] +
                  ["-solutionFile", "stripes.csv"])
        p.communicate()

    tensor = read_2D("stripes.csv")

    animate_colormap(tensor, np.linspace(0.0, 1.0, tensor.shape[2]), np.linspace(0.0, 1.0, tensor.shape[1]), True)


def run_gray_scott_bacteria2(run=True):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "GrayScott"] +
                  ["-boundaryCondition", "Periodic"] +
                  ["-xDimension", "64"] +
                  ["-yDimension", "64"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "1.0"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", ".14"] +
                  ["-vDiffusion", ".06"] +
                  ["-patternParameter1", "0.035"] +
                  ["-patternParameter2", ".065"] +
                  ["-solutionFile", "bacteria2.csv"])
        p.communicate()

    tensor = read_2D("bacteria2.csv")

    animate_colormap(tensor, np.linspace(0.0, 1.0, tensor.shape[2]), np.linspace(0.0, 1.0, tensor.shape[1]), True)


def run_gray_scott_coral(run=True):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "GrayScott"] +
                  ["-boundaryCondition", "Periodic"] +
                  ["-xDimension", "32"] +
                  ["-yDimension", "32"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", ".05"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", ".19"] +
                  ["-vDiffusion", ".05"] +
                  ["-patternParameter1", ".06"] +
                  ["-patternParameter2", ".02"] +
                  ["-solutionFile", "coral.csv"])
        p.communicate()

    tensor = read_2D("coral.csv")

    animate_colormap(tensor, np.linspace(0.0, 1.0, tensor.shape[2]), np.linspace(0.0, 1.0, tensor.shape[1]), True)


def run_gray_scott_lines(run=True):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "GrayScott"] +
                  ["-boundaryCondition", "Periodic"] +
                  ["-xDimension", "64"] +
                  ["-yDimension", "64"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "1"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", ".16"] +
                  ["-vDiffusion", ".08"] +
                  ["-patternParameter1", ".05"] +
                  ["-patternParameter2", ".065"] +
                  ["-solutionFile", "lines.csv"])
        p.communicate()

    tensor = read_2D("lines.csv")

    animate_colormap(tensor, np.linspace(0.0, 1.0, tensor.shape[2]), np.linspace(0.0, 1.0, tensor.shape[1]), True)


def run_brussellator_stripes(run=True):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "Brussellator"] +
                  ["-boundaryCondition", "ZeroFlux"] +
                  ["-xDimension", "64"] +
                  ["-yDimension", "64"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "0.01"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", "2"] +
                  ["-vDiffusion", "16"] +
                  ["-patternParameter1", "4.5"] +
                  ["-patternParameter2", "7.5"] +
                  ["-solutionFile", "br_stripes.csv"])
        p.communicate()

    tensor = read_2D("br_stripes.csv")

    animate_colormap(tensor, np.linspace(0.0, 1.0, tensor.shape[2]), np.linspace(0.0, 1.0, tensor.shape[1]), show=True)


def run_brussellator_dots(run=True):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "Brussellator"] +
                  ["-boundaryCondition", "ZeroFlux"] +
                  ["-xDimension", "64"] +
                  ["-yDimension", "64"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "0.01"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", "2"] +
                  ["-vDiffusion", "16"] +
                  ["-patternParameter1", "4.5"] +
                  ["-patternParameter2", "12"] +
                  ["-solutionFile", "br_dots.csv"])
        p.communicate()

    tensor = read_2D("br_dots.csv")

    animate_colormap(tensor, np.linspace(0.0, 1.0, tensor.shape[2]), np.linspace(0.0, 1.0, tensor.shape[1]), show=True)


if __name__ == "__main__":
    run_brussellator_dots(run=True)