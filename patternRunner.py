import numpy as np
from subprocess import Popen
import os
from plotter import animate_colormap

debugDll = os.getcwd() + "\\x64\\Debug\\TuringPatterns.exe"
releaseDll = os.getcwd() + "\\x64\\Release\\TuringPatterns.exe"


def run_gray_scott_bacteria(run=True, save=False):
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
                  ["-solutionFile", "bacteria.npy"])
        p.communicate()

    tensor = np.load("bacteria.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="bacteria.gif")


def run_gray_scott_bacteria2(run=True, save=False):
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
                  ["-solutionFile", "bacteria2.npy"])
        p.communicate()

    tensor = np.load("bacteria2.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="bacteria2.gif")


def run_gray_scott_coral(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "GrayScott"] +
                  ["-boundaryCondition", "Periodic"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", ".5"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", ".19"] +
                  ["-vDiffusion", ".05"] +
                  ["-patternParameter1", ".06"] +
                  ["-patternParameter2", ".02"] +
                  ["-solutionFile", "coral.npy"])
        p.communicate()

    tensor = np.load("coral.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="coral.gif")


def run_gray_scott_coral2(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "GrayScott"] +
                  ["-boundaryCondition", "ZeroFlux"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", ".5"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", ".19"] +
                  ["-vDiffusion", ".05"] +
                  ["-patternParameter1", ".01"] +
                  ["-patternParameter2", ".015"] +
                  ["-solutionFile", "coral2.npy"])
        p.communicate()

    tensor = np.load("coral2.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="coral2.gif")


def run_gray_scott_coral3(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "GrayScott"] +
                  ["-boundaryCondition", "Periodic"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", ".5"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", ".19"] +
                  ["-vDiffusion", ".05"] +
                  ["-patternParameter1", ".03"] +
                  ["-patternParameter2", ".025"] +
                  ["-solutionFile", "coral3.npy"])
        p.communicate()

    tensor = np.load("coral3.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="coral3.gif")

def run_gray_scott_lines(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "GrayScott"] +
                  ["-boundaryCondition", "Periodic"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "1"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", ".16"] +
                  ["-vDiffusion", ".08"] +
                  ["-patternParameter1", ".05"] +
                  ["-patternParameter2", ".065"] +
                  ["-solutionFile", "lines.npy"])
        p.communicate()

    tensor = np.load("lines.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="lines.gif")


def run_brussellator_stripes(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "Brussellator"] +
                  ["-boundaryCondition", "ZeroFlux"] +
                  ["-xDimension", "256"] +
                  ["-yDimension", "256"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "0.01"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", "2"] +
                  ["-vDiffusion", "16"] +
                  ["-patternParameter1", "4.5"] +
                  ["-patternParameter2", "7.5"] +
                  ["-solutionFile", "br_stripes.npy"])
        p.communicate()

    tensor = np.load("br_stripes.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="br_stripes.gif")


def run_brussellator_dots(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "Brussellator"] +
                  ["-boundaryCondition", "Periodic"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "0.01"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", "2"] +
                  ["-vDiffusion", "16"] +
                  ["-patternParameter1", "4.5"] +
                  ["-patternParameter2", "12"] +
                  ["-solutionFile", "br_dots.npy"])
        p.communicate()

    tensor = np.load("br_dots.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='Spectral',
                     show=not save,
                     save=save,
                     name="br_dots.gif")


def run_schnakenberg(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "Schnakenberg"] +
                  ["-boundaryCondition", "ZeroFlux"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "200"] +
                  ["-dt", "0.01"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", "1.0"] +
                  ["-vDiffusion", "10"] +
                  ["-patternParameter1", ".1"] +
                  ["-patternParameter2", ".9"] +
                  ["-solutionFile", "schnakenberg.npy"])
        p.communicate()

    tensor = np.load("schnakenberg.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='seismic',
                     show=not save,
                     save=save,
                     name="schnakenberg.gif")


def run_thomas(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "Thomas"] +
                  ["-boundaryCondition", "ZeroFlux"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "100"] +
                  ["-dt", "0.0005"] +
                  ["-whiteNoiseScale", ".1"] +
                  ["-uDiffusion", "1.0"] +
                  ["-vDiffusion", "28"] +
                  ["-patternParameter1", "150"] +
                  ["-patternParameter2", "100"] +
                  ["-solutionFile", "thomas.npy"])
        p.communicate()

    tensor = np.load("thomas.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='seismic',
                     show=not save,
                     save=save,
                     name="schnakenberg.gif")


def run_fitz_hugh_nagumo(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "FitzHughNagumo"] +
                  ["-boundaryCondition", "ZeroFlux"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "1000"] +
                  ["-dt", "0.001"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", "1"] +
                  ["-vDiffusion", "100"] +
                  ["-patternParameter1", "-0.005"] +
                  ["-patternParameter2", "10.0"] +
                  ["-solutionFile", "fhn.npy"])
        p.communicate()

    tensor = np.load("fhn.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="fhn.gif")


def run_fitz_hugh_nagumo_low_beta(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "FitzHughNagumo"] +
                  ["-boundaryCondition", "ZeroFlux"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "1000"] +
                  ["-dt", "0.001"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", "1"] +
                  ["-vDiffusion", "100"] +
                  ["-patternParameter1", "0.01"] +
                  ["-patternParameter2", ".25"] +
                  ["-solutionFile", "fhnb.npy"])
        p.communicate()

    tensor = np.load("fhnb.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="fhnb.gif")


def run_fitz_hugh_nagumo_spatial(run=True, save=False):
    if run:
        p = Popen([releaseDll] +
                  ["-pattern", "FitzHughNagumo"] +
                  ["-boundaryCondition", "ZeroFlux"] +
                  ["-xDimension", "128"] +
                  ["-yDimension", "128"] +
                  ["-nIter", "100"] +
                  ["-nIterPerRound", "1000"] +
                  ["-dt", "0.001"] +
                  ["-whiteNoiseScale", ".05"] +
                  ["-uDiffusion", "1"] +
                  ["-vDiffusion", "100"] +
                  ["-patternParameter1", "0.01"] +
                  ["-patternParameter2", "10"] +
                  ["-solutionFile", "fhns.npy"])
        p.communicate()

    tensor = np.load("fhns.npy")

    animate_colormap(tensor,
                     np.linspace(0.0, 1.0, tensor.shape[2]),
                     np.linspace(0.0, 1.0, tensor.shape[1]),
                     cmap='RdBu',
                     show=not save,
                     save=save,
                     name="fhns.gif")


if __name__ == "__main__":
    run_gray_scott_coral3(run=False, save=True)