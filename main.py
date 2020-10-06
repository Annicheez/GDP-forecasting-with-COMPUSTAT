import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn    
import statsmodels.formula.api as smf
import statsmodels.api as sm
from Extraction_script import extract_estimates, seasonal_adjustment


## importing relevant data ---
try:
    super_data = pd.read_csv(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Python-Project\data.csv')
    data_quarterly = pd.read_csv(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Python-Project\data_quarterly.csv')
    additional_data = pd.read_csv(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Python-Project\a4a60a39283592d5.csv')
    inv_data = pd.read_csv(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Python-Project\8507ae5868260590.csv')
    ngdp = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\NOUTPUTQvQd.xlsx')
    npci = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\nconQvQd.xlsx')
    rinvch = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\rinvchiQvQd.xlsx')
    routput = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\ROUTPUTQvQd.xlsx')
    rpci = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\RCONQvQd.xlsx')
    rpdinonres = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\rinvbfQvQd.xlsx')
    rpdires = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\rinvresidQvQd.xlsx')
    cpi = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\cpiQvMd.xlsx')
except:
    super_data = pd.read_csv(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Python-Project\data.csv')
    data_quarterly = pd.read_csv(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Python-Project\data_quarterly.csv')
    additonal_data = pd.read_csv(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Python-Project\a4a60a39283592d5.csv')
    inv_data = pd.read_csv(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Python-Project\8507ae5868260590.csv')
    ngdp = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\NOUTPUTQvQd.xlsx')
    npci = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\nconQvQd.xlsx')
    rinvch = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\rinvchiQvQd.xlsx')
    routput = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\ROUTPUTQvQd.xlsx')
    rpci = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\RCONQvQd.xlsx')
    rpdinonres = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\rinvbfQvQd.xlsx')
    rpdires = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\rinvresidQvQd.xlsx')
    cpi = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\Raw_downloads\cpiQvMd.xlsx')
## Extracting data from Phil-Fed datasets and setting column identifiers##
""" 

As column names are set from the extracted data, there is no feasible way to convert object names to a string.
A generic function could not be written to do this procedure.

"""

## Nominal GDP ##
ngdp = extract_estimates(ngdp)

name_string = pd.Series(['ngdp_'] * (len(ngdp.columns)-2))
estimate_names = name_string.str.cat(pd.Series(ngdp.columns[:len(ngdp.columns)-2])) 
date_names = pd.Series(ngdp.columns[-2:])
ngdp.columns = estimate_names.append(date_names)

## Real GDP ##
rgdp = extract_estimates(routput)

name_string = pd.Series(['rgdp_'] * (len(rgdp.columns)-2))
estimate_names = name_string.str.cat(pd.Series(rgdp.columns[:len(rgdp.columns)-2])) 
date_names = pd.Series(rgdp.columns[-2:])
rgdp.columns = estimate_names.append(date_names)


## Nominal PCI ##
npci = extract_estimates(npci)

name_string = pd.Series(['npci_'] * (len(npci.columns)-2))
estimate_names = name_string.str.cat(pd.Series(npci.columns[:len(npci.columns)-2])) 
date_names = pd.Series(npci.columns[-2:])
npci.columns = estimate_names.append(date_names)

## Real PCI ##
rpci = extract_estimates(rpci)

name_string = pd.Series(['rpci_'] * (len(rpci.columns)-2))
estimate_names = name_string.str.cat(pd.Series(rpci.columns[:len(rpci.columns)-2])) 
date_names = pd.Series(rpci.columns[-2:])
rpci.columns = estimate_names.append(date_names)



## Real INCH ##
rinvch = extract_estimates(rinvch)

name_string = pd.Series(['ninvch_'] * (len(rinvch.columns)-2))
estimate_names = name_string.str.cat(pd.Series(rinvch.columns[:len(rinvch.columns)-2])) 
date_names = pd.Series(rinvch.columns[-2:])
rinvch.columns = estimate_names.append(date_names)

## Real PDI Non-residential ##
rpdinonres = extract_estimates(rpdinonres)

name_string = pd.Series(['npdinonres_'] * (len(rpdinonres.columns)-2))
estimate_names = name_string.str.cat(pd.Series(rpdinonres.columns[:len(rpdinonres.columns)-2])) 
date_names = pd.Series(rpdinonres.columns[-2:])
rpdinonres.columns = estimate_names.append(date_names)


## Real PDI Residential ##
rpdires = extract_estimates(rpdires)

name_string = pd.Series(['npdires_'] * (len(rpdires.columns)-2))
estimate_names = name_string.str.cat(pd.Series(rpdires.columns[:len(rpdires.columns)-2])) 
date_names = pd.Series(rpdires.columns[-2:])
rpdires.columns = estimate_names.append(date_names)

del estimate_names
del date_names
del name_string

## Nominal converstion using CPI ###

cpi_data = pd.DataFrame(cpi.iloc[:,-1])
cpi_data['Year'] = cpi['DATE'].str.split(':',expand = True).iloc[:,0]
cpi_data['Quarter'] = cpi['DATE'].str.split(':',expand = True).iloc[:,1].str.replace('0', 'Q')
cpi_data = cpi_data[cpi_data.Quarter.str.contains('3|6|9|12', regex = True)]
replace = {'Q3': 'Q1', 'Q6': 'Q2', 'Q9': 'Q3', '12': 'Q4'}
cpi_data['Quarter'] = cpi_data.Quarter.map(replace)
cpi_data['Date'] = cpi_data['Year'].str.cat(cpi_data['Quarter'])
cpi_data.reset_index(inplace = True, drop = True)

def to_nominal_cpi(dataset, index_series = cpi_data):
    
    dataset['Date'] = dataset['Year'].str.cat(dataset['Quarter'])
    common_dates = set(dataset['Date']).intersection(set(index_series['Date']))
    dataset_sub = dataset[dataset.Date.isin(common_dates)]
    index_series_sub = index_series[index_series.Date.isin(common_dates)]
    dataset_sub_num = dataset.filter(dataset.columns[dataset.dtypes == 'float64'], axis = 1)
    dataset_sub_date = dataset.filter(dataset.columns[~(dataset.dtypes == 'float64')], axis = 1)
    nominal_dataset_num = dataset_sub_num.reset_index(drop = True).mul(index_series_sub.filter(index_series_sub.columns[index_series_sub.dtypes == 'float64'], axis = 1).reset_index(drop = True)['CPI20Q2'] / 100, axis = 0)
    nominal_dataset = pd.concat([nominal_dataset_num, dataset_sub_date], axis = 1)
    nominal_dataset.drop('Date', inplace = True, axis = 1)
    return nominal_dataset
    
npdinonres = to_nominal_cpi(rpdinonres)
ninvch = to_nominal_cpi(rinvch)
npdires = to_nominal_cpi(rpdires)

### Merging all economic data into a single dataframe ---

economic_data = pd.DataFrame([])
dataframes = [npci, ngdp , ninvch, npdinonres, npdires]

for i, dataframe in enumerate(dataframes):
    if i == 0:  economic_data = dataframe
    else: economic_data = pd.merge(left = economic_data, right = dataframe, on = ['Year', 'Quarter'], how = 'left')

economic_data['Date'] = economic_data.Year.str.cat(economic_data.Quarter.str)
economic_data.set_index(pd.to_datetime(economic_data.Date), drop = True, inplace = True)

## Quarterly dataset cleaning and aggregation---
## Additonal data merging----
additional_data = pd.merge(additional_data, inv_data.filter(['cusip', 'datacqtr', 'invtq']), on = ['datacqtr', 'cusip'], how = 'left')
data_quarterly = pd.merge(data_quarterly, additional_data.filter(['cusip', 'datacqtr', 'intanq', 'ppegtq', 'rectrq', 'xrdq', 'invfgtq']).drop_duplicates(['cusip', 'datacqtr'], keep = 'last'), on = ['cusip', 'datacqtr'], how = 'left')

data_quarterlyc = data_quarterly.filter(['cusip', 'conm', 'curcdq', 'datacqtr', 'rdq', 'ancq', 'cogsq', 'dpq', 
                                         'invtq', 'revtq','tieq', 'tiiq', 'xoprq', 'invchy', 'costat', 'sic',
                                         'intanq', 'ppegtq', 'rectrq', 'xrdq', 'invfgtq'])
data_quarterlyc.loc[data_quarterlyc.revtq < 0, 'revtq'] = 0         ## Removing negative sales
data_quarterlyc = data_quarterlyc[~data_quarterlyc.datacqtr.isna()].drop_duplicates(['cusip', 'datacqtr'], keep = 'last')           ##drop missing observations and duplicates




## Capital Expenditure Variable ---
data_quarterlyc['delta_ppeg'] = data_quarterlyc.groupby('cusip')['ppegtq'].apply(lambda x:x) - data_quarterlyc.groupby('cusip')['ppegtq'].shift(1) 
data_quarterlyc.loc[data_quarterlyc.delta_ppeg < 0, 'delta_ppeg'] = 0
data_quarterlyc['delta_intang'] = data_quarterlyc.groupby('cusip')['intanq'].apply(lambda x:x) - data_quarterlyc.groupby('cusip')['intanq'].shift(1) 
data_quarterlyc.loc[data_quarterlyc.delta_intang < 0, 'delta_intang'] = 0
data_quarterlyc['capex'] = data_quarterlyc['xrdq'].fillna(0) + data_quarterlyc['delta_intang'].fillna(0) + data_quarterlyc['delta_ppeg'].fillna(0) 
data_quarterlyc['capex'].replace(0, np.nan, inplace = True)

## Financial Services adjustment -- implicit financial services

data_quarterlyc['implicit_finservices'] = data_quarterlyc.tiiq - data_quarterlyc.tieq   
data_quarterlyc['revtq'] = data_quarterlyc.implicit_finservices.combine_first(data_quarterlyc.revtq)

## CIPI variable

data_quarterlyc['delta_invq'] = data_quarterlyc.groupby('cusip')['invtq'].apply(lambda x:x) - data_quarterlyc.groupby('cusip')['invtq'].shift(1)

##filter for sic codes ---
try:
    with open(r'C:\\Users\\Anas Khan\\OneDrive - The University of Melbourne\\Desktop\\Projects\\Accounting and Economics\\Python-Project\\siccodes.txt', mode  ='r') as f:
        x = f.readlines()
except:
    with open(r'C:\\Users\\annic\\OneDrive - The University of Melbourne\\Desktop\\Projects\\Accounting and Economics\\Python-Project\\siccodes.txt', mode  ='r') as f:
        x = f.readlines()

sic_codes = [int(word.replace('\n', '')) for word in x]

booleans_quarterly = []
for index, sic_code in data_quarterlyc.sic.iteritems():
    if sic_code in sic_codes:
        booleans_quarterly.append(True)
    else:
        booleans_quarterly.append(False)

data_quarterlyc = data_quarterlyc[booleans_quarterly]


## Aggregatign  by quarter
data_quarterlyq = data_quarterlyc.filter(['datacqtr','rdq', 'ancq', 'cogsq', 'dpq', 
                                         'invtq', 'revtq','tieq', 'tiiq', 'xoprq', 'invchy', 'capex', 'delta_invq']).groupby('datacqtr').sum()
data_quarterlyq.reset_index(inplace = True)
data_quarterlyq.set_index(pd.to_datetime(data_quarterlyq.datacqtr), inplace = True)
data_quarterlyq.index.name = ''

### Seasonally Adjusting Revenue series using Census Bureau X13 Arima model---
data_quarterlyq['revtq'] = seasonal_adjustment(data_quarterlyq.filter(['revtq']))

## Combining final datasets
data_quarterlyqagg = pd.merge(data_quarterlyc, data_quarterlyq, how = 'left', on = 'datacqtr', suffixes = ('','_qagg'))
data_quarterlyqagg[['Year', 'Quarter']]= data_quarterlyqagg.datacqtr.str.split('Q', expand = True)

data_sicclean = data_quarterlyqagg.copy()

## Data for CIPI
inv_data = data_sicclean.filter(['delta_invq_qagg', 'datacqtr']).drop_duplicates('datacqtr', keep = 'last')
inv_data.set_index(pd.to_datetime(inv_data.datacqtr), inplace = True)

bools = economic_data.columns.str.contains('ninvch')
col_names = economic_data.columns[bools]
invch_data = economic_data.filter(col_names) * 1000/4 

reg_data_inv = invch_data.join(inv_data, how = 'left')
reg_data_inv['ninvch_revision_thirdest'] = reg_data_inv['ninvch_advance_estimate'] - reg_data_inv['ninvch_third_estimate'] 
reg_data_inv['ninvch_revision_secondannualest'] = reg_data_inv['ninvch_advance_estimate'] - reg_data_inv['ninvch_second_ann_estimate'] 
reg_data_inv['ninvch_revision_compest'] = reg_data_inv['ninvch_advance_estimate'] - reg_data_inv['ninvch_comp_estimate'] 
reg_data_cipi = reg_data_inv.filter(['delta_invq_qagg', 'ninvch_revision_thirdest', 'ninvch_revision_secondannualest', 'ninvch_revision_compest'])
reg_data_cipi = reg_data_cipi.loc[(reg_data_cipi.index >= '1988-01-01') & (reg_data_cipi.index < '2020-01-01')]

## Data for PFI
bools = economic_data.columns.str.contains('pdinonres')
col_names = economic_data.columns[bools]
pfi_data = economic_data.filter(col_names)
pfi_data.columns = pfi_data.columns.str.replace('npdinonres', 'npfi')   ### Name correction
pfi_data = pfi_data.copy() * 1000/4                                     ### 1000 for coverting to millions of dollars and 4 for quarterly conversion

capex_data = data_sicclean.filter(['capex_qagg', 'datacqtr']).drop_duplicates('datacqtr', keep = 'last').reset_index(drop= True)
capex_data.set_index(pd.to_datetime(capex_data.datacqtr), inplace = True)

reg_data_pfi = pfi_data.join(capex_data, how = 'left')
reg_data_pfi['npfi_revision_thirdest'] = reg_data_pfi['npfi_advance_estimate'] - reg_data_pfi['npfi_third_estimate']
reg_data_pfi['npfi_revision_secondannualest'] = reg_data_pfi['npfi_advance_estimate'] - reg_data_pfi['npfi_second_ann_estimate']
reg_data_pfi['npfi_revision_compest'] = reg_data_pfi['npfi_advance_estimate'] - reg_data_pfi['npfi_comp_estimate']
reg_data_pfi = reg_data_pfi.filter(['npfi_revision_thirdest', 'npfi_revision_secondannualest', 'npfi_revision_compest', 'capex_qagg'])
reg_data_pfi.index.name = 'Date'
reg_data_pfi = reg_data_pfi.loc[(reg_data_pfi.index >= '1988-01-01') & (reg_data_pfi.index < '2020-01-01')]


## Data for PCE
rev_data = data_sicclean.filter(['revtq_qagg', 'datacqtr'])
rev_data.set_index(pd.to_datetime(rev_data.datacqtr), inplace = True)
rev_data = rev_data.drop('datacqtr', axis = 1).drop_duplicates().sort_values('datacqtr')
rev_data.index.name = 'Date'

bools = economic_data.columns.str.contains('npci')
col_names = economic_data.columns[bools]
economic_data_npci = economic_data.filter(col_names) * 1000/4                   ### 1000 for conversion to millions of dollars and 4 for conversion to quarterly

npci_data = economic_data_npci.join(rev_data, on = 'Date', how = 'left')
npci_data['npci_revision_thirdest'] = npci_data['npci_advance_estimate'] - npci_data['npci_third_estimate']
npci_data['npci_revision_secondannualest'] = npci_data['npci_advance_estimate'] - npci_data['npci_second_ann_estimate']
npci_data['npci_revision_compest'] = npci_data['npci_advance_estimate'] - npci_data['npci_comp_estimate']
reg_data_npci = npci_data.filter(['datacqtr', 'npci_revision_thirdest', 'npci_revision_secondannualest', 'npci_revision_compest', 'revtq_qagg'])
reg_data_npci = reg_data_npci[(reg_data_npci.index >= '1988-01-01') & (reg_data_npci.index < '2020-01-01')]


### Loading and cleaning SPF forecast data
try:
    data_rpci_meanspf = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\SPF\Mean_RCONSUM_Level.xlsx') 
    data_resin_meanspf = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\SPF\Mean_RNRESIN_Level.xlsx') 
    data_rresin_meanspf = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\SPF\Mean_RRESINV_Level.xlsx') 
    data_rbi_meanspf = pd.read_excel(r'C:\Users\Anas Khan\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\SPF\Mean_RCBI_Level.xlsx') 

except:
    data_rpci_meanspfr = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\SPF\Mean_RCONSUM_Level.xlsx')
    data_resin_meanspf = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\SPF\Mean_RNRESIN_Level.xlsx') 
    data_rresin_meanspf = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\SPF\Mean_RRESINV_Level.xlsx') 
    data_rbi_meanspf = pd.read_excel(r'C:\Users\annic\OneDrive - The University of Melbourne\Desktop\Projects\Accounting and Economics\Phil. Fed Data\SPF\Mean_RCBI_Level.xlsx') 

data_rpci_meanspf['Date'] = data_rpci_meanspf['YEAR'].astype('str').str.cat(data_rpci_meanspf['QUARTER'].astype('str'), sep = 'Q')
data_rpci_meanspf.set_index(pd.to_datetime(data_rpci_meanspf.Date), drop = True, inplace = True)

data_resin_meanspf['Date'] = data_resin_meanspf['YEAR'].astype('str').str.cat(data_resin_meanspf['QUARTER'].astype('str'), sep = 'Q')
data_resin_meanspf.set_index(pd.to_datetime(data_resin_meanspf.Date), drop = True, inplace = True)

data_rresin_meanspf['Date'] = data_rresin_meanspf['YEAR'].astype('str').str.cat(data_rresin_meanspf['QUARTER'].astype('str'), sep = 'Q')
data_rresin_meanspf.set_index(pd.to_datetime(data_rresin_meanspf.Date), drop = True, inplace = True)

data_rbi_meanspf['Date'] = data_rbi_meanspf['YEAR'].astype('str').str.cat(data_rbi_meanspf['QUARTER'].astype('str'), sep = 'Q')
data_rbi_meanspf.set_index(pd.to_datetime(data_rbi_meanspf.Date), drop = True, inplace = True)


## Convert to nominal
def to_nominal_cpi_spf(dataset, index_series = cpi_data):

    common_dates = set(dataset['Date']).intersection(set(index_series['Date']))
    dataset_sub = dataset[dataset.Date.isin(common_dates)]
    index_series_sub = index_series[index_series.Date.isin(common_dates)]
    dataset_sub_num = dataset.filter(dataset.columns[dataset.dtypes == 'float64'], axis = 1)
    dataset_sub_date = dataset.filter(dataset.columns[~(dataset.dtypes == 'float64')], axis = 1)
    nominal_dataset_num = dataset_sub_num.reset_index(drop = True).mul(index_series_sub.filter(index_series_sub.columns[index_series_sub.dtypes == 'float64'], axis = 1).reset_index(drop = True)['CPI20Q2'] / 100, axis = 0)
    nominal_dataset = pd.concat([nominal_dataset_num, dataset_sub_date.reset_index(drop = True)], axis = 1)
    return nominal_dataset

data_rpci_meanspf = to_nominal_cpi_spf(data_rpci_meanspf)
data_resin_meanspf = to_nominal_cpi_spf(data_resin_meanspf)
data_rresin_meanspf = to_nominal_cpi_spf(data_rresin_meanspf)
data_rbi_meanspf = to_nominal_cpi_spf(data_rbi_meanspf)

## Creating Error variables ##
## PCE
npci_spf_data = data_rpci_meanspf.filter(['RCONSUM1']) * 1000/4
npci_spf_data.set_index(pd.to_datetime(data_rpci_meanspf.Date), inplace = True)
npci_spf_data['spf_forecast_npci'] = npci_spf_data['RCONSUM1'].shift(-1)         ##Shifting because the dataset arrangement contains estimates of the previous quarter to which the survey is conducted
npci_spf_data.drop(['RCONSUM1'], axis = 1, inplace = True)
npci_spfreg_data = pd.merge(npci_spf_data, economic_data_npci, on = 'Date', how = 'inner')

npci_spfreg_data['npci_advanceest_error'] = npci_spfreg_data['spf_forecast_npci'] - npci_spfreg_data['npci_advance_estimate']
npci_spfreg_data['npci_thirdest_error'] = npci_spfreg_data['spf_forecast_npci'] - npci_spfreg_data['npci_third_estimate']
npci_spfreg_data['npci_secondannest_error'] = npci_spfreg_data['spf_forecast_npci'] - npci_spfreg_data['npci_second_ann_estimate']
npci_spfreg_data['npci_compest_error'] = npci_spfreg_data['spf_forecast_npci'] - npci_spfreg_data['npci_comp_estimate']

npci_spfreg_data = pd.merge(npci_spfreg_data, reg_data_npci.filter(['revtq_qagg']), on = 'Date', how = 'left')
npci_spfreg_data = npci_spfreg_data[(npci_spfreg_data.index >= '1988-01-01') & (npci_spfreg_data.index < '2020-01-01')]


##CIPI
cipi_spf_data = data_rbi_meanspf.filter(['RCBI1']) * 1000/4
cipi_spf_data.set_index(pd.to_datetime(data_rbi_meanspf.Date), inplace = True)
cipi_spf_data['spf_forecast_cipi'] = cipi_spf_data.RCBI1.shift(-1)
cipi_spfreg_data = pd.merge(cipi_spf_data, invch_data, on = 'Date', how = 'inner')

cipi_spfreg_data['cipi_advanceest_error'] = cipi_spfreg_data['spf_forecast_cipi'] - cipi_spfreg_data['ninvch_advance_estimate']
cipi_spfreg_data['cipi_thirdest_error'] = cipi_spfreg_data['spf_forecast_cipi'] - cipi_spfreg_data['ninvch_third_estimate']
cipi_spfreg_data['cipi_secondannest_error'] = cipi_spfreg_data['spf_forecast_cipi'] - cipi_spfreg_data['ninvch_second_ann_estimate']
cipi_spfreg_data['cipi_compest_error'] = cipi_spfreg_data['spf_forecast_cipi'] - cipi_spfreg_data['ninvch_comp_estimate']

cipi_spfreg_data = pd.merge(cipi_spfreg_data, reg_data_cipi.filter(['delta_invq_qagg']), on = 'Date', how = 'left')
cipi_spfreg_data = cipi_spfreg_data[(cipi_spfreg_data.index >= '1988-01-01') & (cipi_spfreg_data.index < '2020-01-01')]

## PFI
pfi_spf_data = data_resin_meanspf.filter(['RNRESIN1']) * 1000/4
pfi_spf_data['spf_forecast_npfi'] = pfi_spf_data.RNRESIN1.shift(-1)
pfi_spf_data.set_index(pd.to_datetime(data_resin_meanspf.Date), inplace = True)


pfi_spfreg_data = pd.merge(pfi_spf_data, pfi_data, on = 'Date', how = 'inner')
pfi_spfreg_data['npfi_advanceest_error'] = pfi_spfreg_data['spf_forecast_npfi'] - pfi_spfreg_data['npfi_advance_estimate']
pfi_spfreg_data['npfi_thirdest_error'] = pfi_spfreg_data['spf_forecast_npfi'] - pfi_spfreg_data['npfi_third_estimate']
pfi_spfreg_data['npfi_secondannest_error'] = pfi_spfreg_data['spf_forecast_npfi'] - pfi_spfreg_data['npfi_second_ann_estimate']
pfi_spfreg_data['npfi_compest_error'] = pfi_spfreg_data['spf_forecast_npfi'] - pfi_spfreg_data['npfi_comp_estimate']


pfi_spfreg_data = pd.merge(pfi_spfreg_data, reg_data_pfi.filter(['capex_qagg']), on = 'Date', how = 'left')
pfi_spfreg_data = pfi_spfreg_data[(pfi_spfreg_data.index >= '1988-01-01') & (pfi_spfreg_data.index < '2020-01-01')]


## Adding difference columns
reg_data_npci['revtq_diff'] = reg_data_npci['revtq_qagg'] - reg_data_npci['revtq_qagg'].shift(1)
npci_spfreg_data['revtq_diff'] = npci_spfreg_data['revtq_qagg'] - npci_spfreg_data['revtq_qagg'].shift(1)
reg_data_pfi['capex_diff'] = reg_data_pfi['capex_qagg'] - reg_data_pfi['capex_qagg'].shift(1)
pfi_spfreg_data['capex_diff'] = pfi_spfreg_data['capex_qagg'] - pfi_spfreg_data['capex_qagg'].shift(1)



dataframes = [reg_data_npci, reg_data_pfi, reg_data_cipi, npci_spfreg_data, pfi_spfreg_data, cipi_spfreg_data]


for i, dataframe in enumerate(dataframes):
    if i == 0: super_data = dataframe
    else: super_data = super_data.join(dataframe, how = 'left', rsuffix = '_', lsuffix = '') 

corr_data = super_data.filter(['revtq_qagg', 'capex_qagg', 'delta_invq_qagg','npci_comp_estimate', 'npfi_comp_estimate',
       'ninvch_comp_estimate', 'spf_forecast_npci', 'spf_forecast_npfi', 'spf_forecast_cipi' ])

cpi_data.set_index(pd.to_datetime(cpi_data.Date), inplace = True, drop = True)
cpi_data = cpi_data.filter(['CPI20Q2'])
reg_data_cipi = pd.merge(reg_data_cipi, cpi_data, on = 'Date', how = 'left')

## Combining fixed investment and CIPI into a single measure
reg_data_mod = reg_data_cipi.filter(['ninvch_revision_thirdest',
       'ninvch_revision_secondannualest', 'ninvch_revision_compest','delta_invq_qagg' ])
reg_data_mod.columns = reg_data_pfi.filter(['npfi_revision_thirdest', 'npfi_revision_secondannualest',
       'npfi_revision_compest', 'capex_qagg']).columns
reg_data_comb = reg_data_pfi.filter(['npfi_revision_thirdest', 'npfi_revision_secondannualest',
       'npfi_revision_compest', 'capex_qagg']).add(reg_data_mod)
corr_data['capex_qagg'] = corr_data['capex_qagg'] + corr_data['delta_invq_qagg']
corr_data['npfi_comp_estimate'] = corr_data['npfi_comp_estimate'] + corr_data['ninvch_comp_estimate']
corr_data['spf_forecast_npfi'] = corr_data['spf_forecast_npfi'] + corr_data['spf_forecast_cipi']

pfi_spfreg_data = pfi_spfreg_data.drop('capex_diff', axis = 1)
cipi_spfreg_data.columns = pfi_spfreg_data.columns

comb_spfreg_data = pfi_spfreg_data.add(cipi_spfreg_data)

corr_data = corr_data.filter(['revtq_qagg', 'capex_qagg', 'npci_comp_estimate',
       'npfi_comp_estimate','spf_forecast_npci',
       'spf_forecast_npfi'  ])

reg_data_npci.to_csv('reg1.csv')

reg_data_comb.to_csv('reg2.csv')

npci_spfreg_data.to_csv('reg4.csv')

comb_spfreg_data.to_csv('reg5.csv')

corr_data.to_csv('corr.csv')

"""
## VAR 
data_var = npci_data.filter(['npci_advance_estimate', 'npci_revised_1', 'npci_revised_2',
       'npci_revised_3', 'npci_annual_revised', 'npci_revised_8',
       'npci_latest_estimate', 'revtq_qagg'])


data = npci_data.filter(['npci_advance_estimate', 'revtq_qagg'])
model = sm.tsa.VAR(data)
results = model.fit(maxlags = 5, ic = 'aic')
irf = results.irf(10)
irf.plot()


for i, column in enumerate(regression_data.columns):
    Series = regression_data[column].dropna()
    result = sm.tsa.adfuller(Series)
    print(f'{column}- p-value = {result[1]}')   
    
for column in regression_data.columns.drop('revtq_qagg_diff'):
    res = smf.ols(formula = f'{column} ~ revtq_qagg_diff', data = regression_data).fit()
    print(res.summary())
    beginningtex = '\\documentclass{report}
    \\usepackage{booktabs}
    \\begin{document'
    endtex = "\end{document}"

    f = open(f'myreg{column}.tex', 'w')
    f.write(beginningtex)
    f.write(res.summary().as_latex())
    f.write(endtex)
    f.close()


for column in regression2_data.columns.drop('revtq_qagg'):
    res = smf.ols(formula = f'{column} ~ revtq_qagg', data = regression2_data).fit()
    print(res.summary())

model = smf.ols(formula = 'npci_advance_estimate_q_diff ~ revtq_qagg_diff', data = npci_data).fit()
print(model.summary())

### OLS Regressions

for i, column in enumerate(regression_data.columns.drop('revtq_qagg_diff')):
    data = regression_data.copy()
    print('_'*20, f'{column} Matching years', '_'*20)
    model = smf.ols(formula = f'{column} ~ revtq_qagg_diff', data = data).fit()
    print(model.pvalues[1])
    print(model.rsquared_adj)
    print('_'*20, f'{column} 1 Step Ahead', '_'*20)
    data[column] = data[column].shift(1) 
    model = smf.ols(formula = f'{column} ~ revtq_qagg_diff', data = data).fit()
    print(model.pvalues[1])
    print(model.rsquared_adj)
    print('_'*20, f'{column} 2 Step Ahead', '_'*20)
    data[column] = data[column].shift(1) 
    model = smf.ols(formula = f'{column} ~ revtq_qagg_diff', data = data).fit()
    print(model.pvalues[1])
    print('_'*20, f'{column} 3 Step Ahead', '_'*20)
    print(model.rsquared_adj)
    data[column] = data[column].shift(1) 
    model = smf.ols(formula = f'{column} ~ revtq_qagg_diff', data = data).fit()
    print(model.pvalues[1])
    print(model.rsquared_adj)
    print('_'*20, f'{column} 4 Step Ahead', '_'*20)
    data[column] = data[column].shift(1) 
    model = smf.ols(formula = f'{column} ~ revtq_qagg_diff', data = data).fit()
    print(model.pvalues[1])
    print(model.rsquared_adj)
    
model = smf.ols(formula = 'npci_latest_estimate_diff2 ~ revtq_qagg_diff', data = regression_data).fit()
print(model.summary())



## Converting to nominal and cleaning ##
forecast_data = to_nominal_cpi(data_rpci_meanspfr)
forecast_data['Date'] = forecast_data.YEAR.astype('str').str.cat(forecast_data.QUARTER.astype('str'), sep = 'Q')
forecast_data.set_index(pd.to_datetime(forecast_data.Date), inplace = True)
forecast_data = forecast_data.filter(forecast_data.columns[forecast_data.dtypes == 'float64'])
## SPF regression for PCI ##
reg3_data = forecast_data.join(rev_data, how = 'inner')
reg3_data = reg3_data.filter(['advance_error',
       'onequarter_error', 'annual_error', 'twoyear_error', 'final_error',
       'revtq_qagg'])

reg3_data['revtq_qagg']  = reg3_data.revtq_qagg.diff()

for column in reg3_data.columns.drop('revtq_qagg'):
    res = smf.ols(formula = f'{column} ~ revtq_qagg', data = reg3_data).fit()
    print(res.summary())
    beginningtex = '\\documentclass{report}
    \\usepackage{booktabs}
    \\begin{document}''
    endtex = "\end{document}"

    f = open(f'myreg{column}.tex', 'w')
    f.write(beginningtex)
    f.write(res.summary().as_latex())
    f.write(endtex)
    f.close()


for column in reg_data.columns.drop(['ninvch_advance_estimate', 'ninvch_revised_1', 'ninvch_revised_2',
       'ninvch_revised_3', 'ninvch_annual_revised', 'ninvch_revised_8',
       'ninvch_latest_estimate', 'datacqtr', 'invchy_qagg','datacqtr', 'invchy_qagg']):
    model = smf.ols(formula = f'{column} ~ invchy_qagg', data = reg_data).fit()
    print(model.summary())
    beginningtex = '\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}''
    endtex = "\end{document}"

    f = open(f'myreg{column}.tex', 'w')
    f.write(beginningtex)
    f.write(model.summary().as_latex())
    f.write(endtex)
    f.close()



reg2_data = pfi_data.join(capex_data, how = 'inner').dropna()
reg2_data['npdi_revision_1'] = reg2_data['npdi_advance_estimate'] - reg2_data['npdi_revised_1'] 
reg2_data['npdi_revision_annual'] = reg2_data['npdi_advance_estimate'] - reg2_data['npdi_annual_revised'] 
reg2_data['npdi_revision_2year'] = reg2_data['npdi_advance_estimate'] - reg2_data['npdi_revised_8'] 
reg2_data['npdi_revision_final'] = reg2_data['npdi_advance_estimate'] - reg2_data['npdi_latest_estimate']
reg2_data['npdi_advance_estimate_diff']  = reg2_data['npdi_advance_estimate'].diff()
reg2_data['npdi_revised_1_diff']  = reg2_data['npdi_revised_1'].diff()
reg2_data['npdi_revised_2_diff']  = reg2_data['npdi_revised_2'].diff()
reg2_data['npdi_revised_3_diff']  = reg2_data['npdi_revised_3'].diff()
reg2_data['npdi_annual_revised_diff']  = reg2_data['npdi_annual_revised'].diff()
reg2_data['npdi_revised_8_diff']  = reg2_data['npdi_revised_8'].diff()
reg2_data['npdi_latest_estimate_diff']  = reg2_data['npdi_latest_estimate'].diff()
reg2_data['capex_qagg_diff']  = reg2_data['capex_qagg'].diff()

reg2_data = reg2_data.filter(['npdi_advance_estimate_diff', 'npdi_revised_1_diff',
       'npdi_revised_2_diff', 'npdi_revised_3_diff',
       'npdi_annual_revised_diff', 'npdi_revised_8_diff',
       'npdi_latest_estimate_diff', 'capex_qagg_diff','capex_qagg', 'npdi_revision_1',
       'npdi_revision_annual', 'npdi_revision_2year', 'npdi_revision_final'])



for i, column in enumerate(reg2_data.columns.drop(['datacqtr'])):
    column_name = column + '_diff' 
    reg2_data[column_name] = reg2_data[column].diff()


diff2_cols = reg2_data.columns[reg2_data.columns.str.contains('revision')]
reg2_data = reg2_data.filter(['npdi_revision_1', 'npdi_revision_annual', 'npdi_revision_2year',
       'npdi_revision_final', 'capex_qagg'])

for column in reg2_data.columns.drop(['capex_qagg']):
    model = smf.ols(formula = f'{column} ~ capex_qagg', data = reg2_data).fit()
    print(model.summary())
    beginningtex = '\\documentclass{report}
\\usepackage{booktabs}
\\begin{document''
    endtex = "\end{document}"

    f = open(f'myreg{column}.tex', 'w')
    f.write(beginningtex)
    f.write(model.summary().as_latex())
    f.write(endtex)
    f.close()
    

reg_data_g = reg_data.drop('datacqtr', axis = 1)
granger_1 = grangers_causation_matrix(reg_data_g, variables = reg_data_g.columns).iloc[:,-1]
reg2_data_g = reg2_data.drop('datacqtr', axis = 1)
granger_2 = grangers_causation_matrix(reg2_data_g, variables = reg2_data_g.columns).iloc[:-1]
"""