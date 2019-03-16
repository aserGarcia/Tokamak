# TOKAMAK

The purpose for the tokamak project is to explore areas of nuclear fusion with the use of plasma.
In particular a tokamak creates plasma from Deuterium and Tritium gases which then collide
experiencing fusion. In turn heat is released and can be used to create steam to power electric turbines.

![Tokamak Diagram](/assets/tokamak.jpg "Tokamak Diagram")

## Language
The main language in this project is Cuda to take advantage of it's parallel programming capabilities.

## Getting Started
To start, clone the repository
```
git clone https://github.com/aserGarcia/Tokamak.git
```

### Prerequisites

Ensure that you have [OpenGL](http://www.prinmath.com/csci5229/misc/install.html) installed and properly set up. The link takes to a page with proper instructions.

### Instructions
To compile the source code, run the Makefile in the root directory
```
make -f Makefile
```
It will make an executable 'output' ; run as below
```
./output
```