import pandas as pd
import numpy as np
import scipy.stats as stats
import re

# We calculate the win/loss ratio's correlation with the population of the city it is in for the **NHL** using **2018** data.

nhl_df=pd.read_csv("~/Downloads/nhl.csv")
cities=pd.read_html("~/Downloads/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
columns_to_keep=['team', 'W', 'L']
nhl_df=nhl_df[columns_to_keep]
indeces_to_keep=[1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,27,28,
                 29,30,31,32,33,34,35,36]
nhl_df=nhl_df.loc[indeces_to_keep]
nhl_df=nhl_df.drop(labels=[35,36], axis=0)
nhl_df['team'].replace('\*', '', regex=True, inplace=True)
nhl_df.reset_index(inplace=True)
nhl_df=nhl_df.drop('index',axis=1)
def splitteam(row):
    row['Team']=row['team'].split(" ")[-1]
    row['Metropolitan area']=row['team'].split(" ")[0]
    return row

nhl_df=nhl_df.apply(splitteam, axis='columns')
nhl_df=nhl_df.drop('team', axis=1)
nhl_df.replace({"New": "New York City", "Tampa": "Tampa Bay Area", "St.": "St. Louis",
                "Vegas": "Las Vegas", "San": "San Francisco Bay Area",
                "Los": "Los Angeles", "Dallas": "Dallas–Fort Worth",
                "Colorado":"Denver", "Washington": "Washington, D.C.",
                "Carolina": "Raleigh", "Florida": "Miami–Fort Lauderdale",
                "Minnesota": "Minneapolis–Saint Paul", "Anaheim": "Los Angeles",
                "Arizona": "Phoenix"},
               inplace=True)
nhl_df[['W','L']]=nhl_df[['W','L']].astype(float)
nhl=nhl_df.groupby('Metropolitan area').agg({'W': np.sum, 'L': np.sum})
nhl.reset_index(inplace=True)
full_nhl=pd.merge(cities, nhl, how='outer', on='Metropolitan area')
full_nhl['win/loss ratio']=full_nhl['W']/(full_nhl['L']+full_nhl['W'])
full_nhl=full_nhl.drop(['NFL','NBA','MLB'], axis=1)
full_nhl=full_nhl[full_nhl['win/loss ratio'].notna()]
full_nhl.rename(columns={'Population (2016 est.)[8]': 'Population'}, inplace=True)

population_by_region = np.float64(full_nhl['Population']) # pass in metropolitan area population from cities
win_loss_by_region = full_nhl['win/loss ratio'] # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

print(stats.pearsonr(population_by_region, win_loss_by_region)[0])


# We calculate the win/loss ratio's correlation with the population of the city it is in for the **NBA** using **2018** data.


nba_df=pd.read_csv("~/Downloads/nba.csv")
cities=pd.read_html("~/Downloads/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
columns_to_keep=['team', 'W', 'L']
nba_df=nba_df[columns_to_keep]
indeces_to_keep=list(range(30))
nba_df=nba_df.loc[indeces_to_keep]
nba_df['team'].replace('\*[ ].*', '', regex=True, inplace=True)
nba_df['team'].replace('[ ]\(.*', '', regex=True, inplace=True)
nba_df.reset_index(inplace=True)
nba_df=nba_df.drop('index',axis=1)
def splitteam2(row):
    row['Team']=row['team'].split(" ")[-1]
    if len(row['team'].split(" "))==2:
        row['Metropolitan area']=row['team'].split(" ")[0]
    else:
        row['Metropolitan area']=row['team'].split(" ")[0]+' '+row['team'].split(" ")[1]
    return row

nba_df=nba_df.apply(splitteam2, axis='columns')
nba_df=nba_df.drop('team', axis=1)
nba_df.replace({"New York": "New York City", "Brooklyn": "New York City",
                "Golden State": "San Francisco Bay Area", "Portland Trail": "Portland",
                "Dallas": "Dallas–Fort Worth", "Washington": "Washington, D.C.",
                "Miami": "Miami–Fort Lauderdale", "Minnesota": "Minneapolis–Saint Paul",
                "Indiana": "Indianapolis", "Utah": "Salt Lake City"},
               inplace=True)
nba_df[['W','L']]=nba_df[['W','L']].astype(float)
nba=nba_df.groupby('Metropolitan area').agg({'W': np.sum, 'L': np.sum})
nba.reset_index(inplace=True)
full_nba=pd.merge(cities, nba, how='left', on='Metropolitan area')
full_nba['win/loss ratio']=full_nba['W']/(full_nba['L']+full_nba['W'])
full_nba=full_nba.drop(['NFL','NHL','MLB'], axis=1)
full_nba=full_nba[full_nba['win/loss ratio'].notna()]
full_nba.rename(columns={'Population (2016 est.)[8]': 'Population'}, inplace=True)

population_by_region = np.float64(full_nba['Population']) # pass in metropolitan area population from cities
win_loss_by_region = full_nba['win/loss ratio'] # pass in win/loss ratio from nba_df in the same order as cities["Metropolitan area"]

print(stats.pearsonr(population_by_region, win_loss_by_region)[0])

#We calculate the win/loss ratio's correlation with the population of the city it is in for the **MLB** using **2018** data.

mlb_df=pd.read_csv("~/Downloads/mlb.csv")
cities=pd.read_html("~/Downloads/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
columns_to_keep=['team', 'W', 'L']
mlb_df=mlb_df[columns_to_keep]
indeces_to_keep=list(range(30))
mlb_df=mlb_df.loc[indeces_to_keep]
def splitteam3(row):
    row['Team']=row['team'].split(" ")[-1]
    if len(row['team'].split(" "))==2:
        row['Metropolitan area']=row['team'].split(" ")[0]
    else:
        row['Metropolitan area']=row['team'].split(" ")[0]+' '+row['team'].split(" ")[1]
    return row

mlb_df=mlb_df.apply(splitteam3, axis='columns')
mlb_df=mlb_df.drop('team', axis=1)
mlb_df.replace({"New York": "New York City", "Boston Red": "Boston",
                "San Francisco": "San Francisco Bay Area",
                "Tampa Bay": "Tampa Bay Area", "Toronto Blue": "Toronto",
                "Washington": "Washington, D.C.", "Chicago White": "Chicago",
                "Miami": "Miami–Fort Lauderdale", "Minnesota": "Minneapolis–Saint Paul",
                "Arizona": "Phoenix", "Colorado":"Denver",
                "Oakland": "San Francisco Bay Area", "Texas": "Dallas–Fort Worth"},
               inplace=True)
mlb_df[['W','L']]=mlb_df[['W','L']].astype(float)
mlb=mlb_df.groupby('Metropolitan area').agg({'W': np.sum, 'L': np.sum})
mlb.reset_index(inplace=True)
full_mlb=pd.merge(cities, mlb, how='left', on='Metropolitan area')
full_mlb['win/loss ratio']=full_mlb['W']/(full_mlb['L']+full_mlb['W'])
full_mlb=full_mlb.drop(['NFL','NHL','NBA'], axis=1)
full_mlb=full_mlb[full_mlb['win/loss ratio'].notna()]
full_mlb.rename(columns={'Population (2016 est.)[8]': 'Population'}, inplace=True)

population_by_region = np.float64(full_mlb['Population']) # pass in metropolitan area population from cities
win_loss_by_region = full_mlb['win/loss ratio'] # pass in win/loss ratio from nba_df in the same order as cities["Metropolitan area"]

print(stats.pearsonr(population_by_region, win_loss_by_region)[0])

#We calculate the win/loss ratio's correlation with the population of the city it is in for the **NFL** using **2018** data.

nfl_df=pd.read_csv("~/Downloads/nfl.csv")
cities=pd.read_html("~/Downloads/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
columns_to_keep=['team', 'W', 'L']
nfl_df=nfl_df[columns_to_keep]
indeces_to_keep=[1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24,26,27,28,
                 29,31,32,33,34,36,37,38,39]
nfl_df=nfl_df.loc[indeces_to_keep]
nfl_df['team'].replace('\*', '', regex=True, inplace=True)
nfl_df['team'].replace('\+', '', regex=True, inplace=True)
nfl_df.reset_index(inplace=True)
nfl_df=nfl_df.drop('index',axis=1)
def splitteam4(row):
    row['Team']=row['team'].split(" ")[-1]
    if len(row['team'].split(" "))==2:
        row['Metropolitan area']=row['team'].split(" ")[0]
    else:
        row['Metropolitan area']=row['team'].split(" ")[0]+' '+row['team'].split(" ")[1]
    return row

nfl_df=nfl_df.apply(splitteam4, axis='columns')
nfl_df=nfl_df.drop('team', axis=1)
nfl_df.replace({"New York": "New York City", "New England": "Boston",
                "San Francisco": "San Francisco Bay Area",
                "Tampa Bay": "Tampa Bay Area", "Tennessee": "Nashville",
                "Washington": "Washington, D.C.", "Chicago White": "Chicago",
                "Miami": "Miami–Fort Lauderdale", "Minnesota": "Minneapolis–Saint Paul",
                "Arizona": "Phoenix", "Carolina": "Raleigh",
                "Oakland": "San Francisco Bay Area", "Dallas": "Dallas–Fort Worth"},
               inplace=True)
nfl_df[['W','L']]=nfl_df[['W','L']].astype(float)
nfl=nfl_df.groupby('Metropolitan area').agg({'W': np.sum, 'L': np.sum})
nfl.reset_index(inplace=True)
full_nfl=pd.merge(cities, nfl, how='outer', on='Metropolitan area')
full_nfl['win/loss ratio']=full_nfl['W']/(full_nfl['L']+full_nfl['W'])
full_nfl=full_nfl.drop(['NHL','NBA','MLB'], axis=1)
full_nfl=full_nfl[full_nfl['win/loss ratio'].notna()]
full_nfl.rename(columns={'Population (2016 est.)[8]': 'Population'}, inplace=True)

population_by_region = np.float64(full_nfl['Population']) # pass in metropolitan area population from cities
win_loss_by_region = full_nfl['win/loss ratio'] # pass in win/loss ratio from nba_df in the same order as cities["Metropolitan area"]

print(stats.pearsonr(population_by_region, win_loss_by_region)[0])
