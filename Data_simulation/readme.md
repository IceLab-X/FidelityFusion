# Data_simulation
Under this folder are files containing data related to the model

## Cost_Function
The calculation method for exploring the cost of data between the main storage fidelity
- cost_currin, cost_park representing the true exploration cost of these two datasets
- cost_pow_10,cost_linear,cost_log represent three different representations of exploration costs between fidelity 10^t, 5*t and log2(t)
## Real_Application
The simulation file that stores real data is called in the same way as the simulation data, but requires the use of Matlab
- HeatedBlock
- VibratePlate
## Synthetic_MF_Function
11 multi-fidelity simulation data, which can be set in continuous or discrete form
| Index | Simulation Name        | Dimensions |
|-------|------------------------|------------|
| 1     | Bohachevsky             | 2-dim      |
| 2     | Booth                   | 2-dim      |
| 3     | Branin                  | 2-dim      |
| 4     | Colvile                 | 4-dim      |
| 5     | Currin                  | 2-dim      |
| 6     | Forrester               | 1-dim      |
| 7     | Hartman                 | 6-dim      |
| 8     | Himmelblau              | 2-dim      |
| 9     | Non-linear Sin          | 1-dim      |
| 10    | Park                    | 2-dim      |
| 11    | Six-hump Camelback      | 2-dim      |
