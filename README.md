# Turing Patterns: A C++/CUDA application for simulating 2D reaction-diffusion dynamics showing diffusion driven instability

This repository implements the numerical schemes for simulating the most popular reaction diffusion dynamics that exhibits Turing instability. 

The general two-dimensional PDE is of the form:

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{array}{rcl}&space;du_t&space;&=&&space;\Delta&space;u&space;&plus;&space;f(u,&space;v)\\&space;dv_t&space;&=&&space;d\Delta&space;v&space;&plus;&space;g(u,&space;v)&space;\end{array}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{array}{rcl}&space;du_t&space;&=&&space;\Delta&space;u&space;&plus;&space;f(u,&space;v)\\&space;dv_t&space;&=&&space;d\Delta&space;v&space;&plus;&space;g(u,&space;v)&space;\end{array}" title="\begin{array}{rcl} du_t &=& \Delta u + f(u, v)\\ dv_t &=& d\Delta v + g(u, v) \end{array}" /></a>

The numerical scheme implemented is a simple Forward Euler scheme, which is known to be unstable but easy to implement. As opposed to the usual stability condition, the introduciton of a source term makes it more unstable, although dt in the region of dx^2 seem to be a reasonable choice. An extension to this could be a Runge-Kutta solver or an IMEX scheme, where the source term is considered in the explicit portion.

The Finite Difference schemes are implemented in CUDA. Since the non-linear nature of the dynamics I found it easier to implement without making use of CuBLAS.

The dynamics that I've implemented are:
- Gray-Scott
- Brussellator
- Schnakernberg
- Thomas
- Fitz-Hugh Nagumo

I made a python script that calls the C++ executable and gather the numerical results and produce the color plot animation. I saved the resulting GIFs, which are shown below:

<p align="center">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/bacteria_compressed.gif">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/bacteria2_compressed.gif">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/br_stripes_compressed.gif">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/coral_compressed.gif">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/coral2_compressed.gif">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/coral3_compressed.gif">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/fhn_compressed.gif">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/fhnb_compressed.gif">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/lines_compressed.gif">
  <img src="https://raw.githubusercontent.com/pmontalb/TuringPatterns/master/schnakenberg_compressed.gif">
</p>

## Dependencies

This code requires CUDA version 11. Download it here:

https://developer.nvidia.com/cuda-11.0-download-archive


## Building

```
git submodule update --init --recursive
mkdir build
cd build/
cmake ..
make

```

# Running

From the top level:

```
python3 patternRunner.py
```
