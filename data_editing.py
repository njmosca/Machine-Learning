import pandas 
import sklearn
import numpy as np
import matplotlib.pyplot as plt 



file = "heart_failure_clinical_records_dataset.csv"
#All Data
data = pandas.read_csv(file)

age = data['age']
full_age_mean = data['age'].mean()
full_age_std = np.std(data['age'])



#Death total
death_samples = data[(data['age'].all() and (data['DEATH_EVENT'] ==1))]
death_age = death_samples['age'] # death age
death_sc = death_samples['serum_creatinine']
death_ef = death_samples['ejection_fraction']


# Living total
living_samples = data[(data['age'].all() and (data['DEATH_EVENT'] ==0))]
living_sc = living_samples['serum_creatinine']
living_ef = living_samples['ejection_fraction']





