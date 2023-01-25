# pdpythonRunAtHome
The quick, run-at-home version of the pdpython repo, generated Jan 2023. This is separate from the development repo, and has been tested on Python 3.10.4 with the library packages listed in requirements.txt.

To run using a Windows terminal virtual environment (macOS works differently, please see [the documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)):

- Download the whole code folder
- cd into the pdpython_model folder
- Create a virtual python environment using each of these commands in turn:

py -m venv env

.\env\Scripts\activate

py -m pip install -r requirements.txt

- Then run any script you require using:

python filename.py

------------------------------------------------------------------------------------------------------------------------------------------------

The main four scripts to run include:

- moody_run.py, a script that runs the 2D grid version of the simulation (no partner swapping), WITH the visual demonstration and easy parameter input
- batch_run.py, a script that runs the 2D grid version of the simulation with hard-coded parameter settings (no partner swapping), WITHOUT visual demo
- fixed_random_run.py, a script that runs the random graph version of the simulation (with partner swapping), WITH the visual demonstration and easy parameter input
- fixed_random_batchrun.py, a script that runs the random graph version of the simulation (with partner swapping), WITHOUT the visual demo

For more detailed instruction, please consult the How to Run .docx/.pdf file contained within.
