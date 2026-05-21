import pandas as pd
import numpy as np

data = pd.read_csv("./PSO_CEC2014_Results.csv").values

best_cost = np.array(data[:, 1])
pd.DataFrame(best_cost).to_excel("sss.xlsx")