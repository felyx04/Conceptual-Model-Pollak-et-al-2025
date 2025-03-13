# Conceptual-Model-Pollak-et-al-2025
This repository contains all the source code of the models used in Pollak et al. 2025. It also includes all the code and data to generate the figures in this paper.

# Structure 
1. **Figures**
    - **../Data**: contains all the data of the individual simulation runs (stored in .pkl format) to re-create the figures in the paper
    - **../Figures_Pollak_2025.ipynb**: Jupyter Notebook to create these figures
    - **../tol_colors.py**: colour-blind safe colour scheme used for the plots. Copyright (c) 2022, Paul Tol
    - **../Plots**: folder contains all figures and supplementary figures in paper

2. **ConceptualModel**
   - **ConceptualModel/Data**: contains the following input files for the conceptual models
     - **../Berends_etal_2020_CP_supplement.dat**: sea-level reconstruction by Berends et al. (2021). Used as the standard target data in the model. Resolution: 100 yr. Paper: https://doi.org/10.5194/cp-17-361-2021. Data: https://doi.org/10.5281/zenodo.3793592
     - **../Data summary sheet Rohling et al_Reviews of Geophysics 2022-v2.xlsx**: comparison and synthesis of sea-level data by Rohling et al. (2022). Resolution: 1 kyr. Paper: https://doi.org/10.1029/2022RG000775. Data: https://doi.org/10.6084/m9.figshare.21430731.v3
     - **../Orbital_Params_-3,6MA-2MA_1kyr_steps.txt**: orbital parameters used as input for the model: obliquity, precession and co-precession obtained from Laskar et al. (2004) solution. Resolution: 1 kyr. Paper: http://dx.doi.org/10.1051/0004-6361:20041335. Web-interface to download data: https://vo.imcce.fr/insola/earth/online/earth/online/index.php

      - **ConceptualModel/Model**: contains the following conceptual models
        - **../ORB3_jax.py**: code for tuning the orbital model (**ORB**). Outputs tuned parameters
        - **../ORB3_jax_plot.py**: code for plotting the orbital model (**ORB**). Requires tuned parameters as input
        - **../ABR4_jax.py**: code for tuning the abrupt model (**ABR**). Outputs tuned parameters
        - **../ABR4_jax_plot.py**: code for plotting the abrupt model (**ABR**). Requires tuned parameters as input
        - **../GRAD2_jax.py**: code for tuning the gradual model (**GRAD**). Outputs tuned parameters
        - **../GRAD2_jax_plot.py**: code for plotting the gradual model (**GRAD**). Requires tuned parameters as input 
        - **../RAMP7_jax.py**: code for tuning the ramp-like model (**RAMP**). Outputs tuned parameters
        - **../RAMP7_jax_plot.py**: code for plotting the ramp-like model (**RAMP**). Requires tuned parameters as input
        - **../RAMP23_jax.py**: code for tuning the ramp-l model (**RAMP-l**). Outputs tuned parameters
        - **../RAMP23_jax_plot.py**: code for plotting the ramp-l model (**RAMP-l**). Requires tuned parameters as input 
