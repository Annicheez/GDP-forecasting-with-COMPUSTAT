import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.tsa.stattools import grangercausalitytests


def extract_estimates(dataframe):
    """ A generic function for extracting relevant estimates from Phil-fed datasets.
    This is a flexible implementation with possibility of customisation as per requirements
    
    returns: a dataframe containing relevant estimates
    """
    estimate_1 = []
    estimate_2 = []
    estimate_3 = []
    estimate_4 = []
    estimate_final = []
    
    starting_index = len(dataframe.iloc[:,1]) - dataframe.iloc[:,1].isna().sum() - 1
    date = dataframe.DATE[starting_index:]
    for index in range(starting_index,len(dataframe)):
        limit = dataframe.iloc[index,:].isna().sum() + 1
        col_ref = dataframe.iloc[index,:limit].isna().sum() + 1
        try:
            estimate_1.append(dataframe.iloc[index, col_ref])
        except:
            estimate_1.append(None)
        try:    
            estimate_2.append(dataframe.iloc[index, col_ref+2])
        except:
            estimate_2.append(None)
        try:
            estimate_3.append(dataframe.iloc[index, col_ref+5])
        except:
            estimate_3.append(None)
        try:
            estimate_4.append(dataframe.iloc[index, col_ref+20])
        except:
            estimate_4.append(None)
        try:
            estimate_final.append(dataframe.iloc[index, len(dataframe.columns)-1])
        except:
            estimate_final.append(None)
            
    extracted_dataframe = pd.DataFrame({'Date': date, 'advance_estimate' : estimate_1, 'third_estimate': estimate_2, 'second_ann_estimate': estimate_3, 'comp_estimate': estimate_4, 'latest_estimate': estimate_final})
    extracted_dataframe.reset_index(drop= True, inplace = True)
    extracted_dataframe[['Year', 'Quarter']] = extracted_dataframe.Date.str.split(":", expand = True)
    extracted_dataframe.drop('Date', axis = 1, inplace = True)
    return extracted_dataframe    

def seasonal_adjustment(dataset):
    """This function seasonally adjusts all columns of a dataset using the X13 arima
    software provided publicly by the Census Bureau
    
    returns: a dataset of seasonally adjusted columns
    """
    for column in dataset.filter(dataset.columns[dataset.dtypes == 'float64']):
        try:
            res = x13_arima_analysis(dataset[column])
            dataset[column] = res.seasadj
        except: pass
    return dataset


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    maxlag = 12
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

