# PyADM1ODE
Implementation of the Anaerobic Digestion Model No. 1 (ADM1) in Python as system of ODEs without any differential algebraic equations. 

The input stream of the ADM1 is calculated from agricultural substrates, so this implementation is especially useful for simulating agricultural co-digestion plants. 

# Installation

Clone or download the repository first.

If you use Anaconda or Miniconda:

    conda env create -f environment.yml

Otherwise (maybe create a virtual environment first):

    pip install -r requirements.txt

Hint: Both files are not tested, so it might be that one or two packages are missing in the files. If so, please write an issue. Thanks!

# Usage

Run main.py

    python main.py

# References
This project is based on https://github.com/CaptainFerMag/PyADM1
