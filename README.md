# DFT-QM_Support_Scripts
Python and bash scripts used to run quantum mechanical calculations, including density functional theory (DFT), via Gaussian16, ORCA6, and xTB-CREST.


**CheckGauss.py:** Used for checking Gaussian16 and ORCA output files, reading a template file and using it to either run the next calculation step of the process or will troubleshoot failed jobs.

**Gauss_out2in.py:** Used in conjunction with CheckGauss.py to construct new Gaussian16 input files from Gaussian16 output files. Reads files ending in CO as constrained optimization, and will iteratively run TS calculation and then SP. For failed jobs it will describe the error type and provide the option to change the keywords before resubmitting. Files with ONIOM partitioning, pseudopotentials, or custom implicit solvation are all supported
