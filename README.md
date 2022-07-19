# CS 471 HW6
## Thomas Bowidowicz, Mike Adams, F21

- The homework Latex report labeled homework6.pdf is found in report subdirectory of the hw6_project directory.
- There are three subdirectories in the hw6 directory:
    1. code - This subdirectory contains all of the source code used in the project and is where the results can be recreated
    2. files - This subdriectory contains all of the output from the code including the plots and output files for the scaling studies.
    3. report - This subdirectory contains the final Latex report

- To recreate the plots found in the report, follow the following instructions for each question:

### Task 1
- To recreate small 8x8 spatial grid mentioned in the report for the serial version of the code, execute the following:
    - Navigate to the code directory located: /hw6_project/code/
    - Execute the following command in the command prompt: python hw6_serialsmall.py
    - The error will be printed to the console.

- To recreate the convergence study plot in the report for the serial version of the code, execute the following:
    - Navigate to the code directory located: /hw6_project/code/
    - Execute the following command in the command prompt: python hw6_serial.py
    - The plot will appear in the code directory

### Task 2 and 3
- To recreate the parallel scaling studies and general execution of the parallel code, execute the following:
    - For the general execution of the parallel code, navigate to the following: /hw6_project/code/
    - Execute the following command in the command prompt: python hw6_parallel.py
    - This will execute the parallel code for the general convergence studies listed in the report.
    - This was carried out on CARC Wheeler with the following command: qsub -q debug /code/hw6_test.pbs
    - The plots for error convergence, initial condition, approximate final solution, and exact final solution will exist in the directory where the pbs script was submitted.

- To recreate the parallel weak scaling study, execute the following:
    - For the weak scaling study of the parallel code, navigate to the following: /hw6_project/code/
    - Execute the following command in the command prompt: python hw6_parallel_weak_#.py (where # is 1, 4, 16, or 64 corresponding to the number of processors)
    - This was carried out on CARC Wheeler with the following command: qsub -q default /code/hw6_weak.pbs
    - This will execute the weak scaling study and create an output file in the hw6_project directory. 
    - To generate the weak scaling study plot as seen in the report, execute the following command: /code/plottimings_weak.py outputFileName.txt
    - The plot for the weak scaling study will exist in the directory where the pbs script was submitted.

- To recreate the parallel strong scaling study, execute the following:
    - For the strong scaling study of the parallel code, navigate to the following: /hw6_project/code/
    - Execute the following command in the command prompt: python hw6_parallelstrong.py
    - This was carried out on CARC Wheeler with the following command: qsub -q default /code/hw6_strong.pbs
    - This will execute the strong scaling study and create an output file in the hw6_project directory. 
    - To generate the strong scaling study plot as seen in the report, execute the following command: /code/plottimings_strong.py outputFileName.txt
    - The plot for the strong scaling study will exist in the directory where the pbs script was submitted.
