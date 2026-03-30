# Conceptual-Model-Pollak-et-al-2025
This repository contains all the source code of the models used in "_Simulating global ice volume across the Mid-Pleistocene Transition with a ramp-like increase in the deglaciation threshold_" (Pollak et al. 2026, https://doi.org/10.5194/cp-22-675-2026). It also includes all the code and data to generate the figures in this paper.

# Structure 
1. **Figures**
    - **../Data**: contains all the data of the individual simulation runs (stored in .pkl format) to re-create the figures in the paper
    - **../Figures_Pollak_2025.ipynb**: Jupyter Notebook to create these figures
    - **../tol_colors.py**: colour-blind safe colour scheme used for the plots. Copyright (c) 2022, Paul Tol
    - **../Plots**: folder contains all figures and supplementary figures in paper

2. **ConceptualModel**
   - **ConceptualModel/Data**: contains the following input files for the conceptual models
     - **../Berends_etal_2020_CP_supplement.dat**: sea-level reconstruction by Berends et al. (2021). Used as one of the two targets for the RAMP model. Resolution: 100 yr. Paper: https://doi.org/10.5194/cp-17-361-2021. Data: https://doi.org/10.5281/zenodo.3793592
     - **../Data summary sheet Rohling et al_Reviews of Geophysics 2022-v2.xlsx**: comparison and synthesis of sea-level data by Rohling et al. (2022). Used as one of the two targets for the RAMP model. Resolution: 1 kyr. Paper: https://doi.org/10.1029/2022RG000775. Data: https://doi.org/10.6084/m9.figshare.21430731.v3
     - **../Orbital_Params_-3,6MA-2MA_1kyr_steps.txt**: orbital parameters used as input for the RAMP model: obliquity, precession and co-precession obtained from Laskar et al. (2004) solution. Resolution: 1 kyr. Paper: https://doi.org/10.1051/0004-6361:20041335. Web interface to download data: https://vo.imcce.fr/insola/earth/online/earth/online/index.php
     - **../Clark_2025_GMSL.xlsx**: Clark 2025 GMSL reconstruction. Used as an additional target for the RAMP-2 model. Paper: https://doi.org/10.1126/science.adv8389  

      - **ConceptualModel/Model**: contains the source code of the new RAMP model presented in Pollak et al. (2025):
        - **../RAMP.py**: code for tuning the RAMP model. Outputs the tuned parameters
        - **../RAMP_plot.py**: code for plotting the RAMP model. Requires the tuned parameters as input
        - **../RAMP-2.py**: code for tuning the RAMP-2 model. Outputs the tuned parameters
        - **../RAMP-2_plot.py**: code for plotting the RAMP-2 model. Requires the tuned parameters as input

3. **environment.yml**
   - contains the conda environment used to run all the code
   - **Note**: not all packages are necessary, but this is my default environment, that's why there are many additional packages
   - **Usage**: you can create a new conda environment from this by using `conda env create -f environment.yml` 

# Citation
If you use this code, please cite:

Pollak et al. (2026). *felyx04/Conceptual-Model-Pollak-et-al-2025*. Zenodo. https://doi.org/10.5281/zenodo.17189824

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17189824.svg)](https://doi.org/10.5281/zenodo.17189824)
