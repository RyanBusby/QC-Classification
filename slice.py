import numpy as np
import pandas as pd

df = pd.read_csv('data/train_numeric.csv')
df.set_index('Id', drop=True, inplace=True)

df_l = df

df = df.drop('Response', axis=1)

#make list of sets of row indices that are not null for every column
#the index of the set in 'ri' corresponds to the index of the column in 'df'
ri = []
for x in xrange(df.shape[1]):
    ri.append(set(np.where(df.iloc[:,x].notnull())[0].tolist()))

#make square matrix and fill with number of shared rows between columns
mat = np.zeros(df.shape[1]**2).reshape(df.shape[1], df.shape[1])

for a in xrange(mat.shape[0]): #this takes a long time to execute
    for b in xrange(a+1, mat.shape[0]):
        print a,b
        mat[a,b] = len(ri[a].intersection(ri[b]))

np.save('data/numpy/similair_columns.npy', mat)

#locate indices of highest scored column pairs and group them together
keys = set(range(df.shape[1]))
groups = []
while len(keys) > 0:
    a = np.where(mat == mat.max())
    a = np.vstack((a[0],a[1])).T
    a.sort()
    for tup in a:
        if tup[0] in keys and tup[1] in keys:
            groups.append({tup[0], tup[1]})
            keys.difference_update([tup[0],tup[1]])
        elif tup[0] in keys and tup[1] not in keys:
            for group in groups:
                if tup[1] in group:
                    group.add(tup[0])
                    break
            keys.remove(tup[0])
        elif tup[0] not in keys and tup[1] in keys:
            for group in groups:
                if tup[0] in group:
                    group.add(tup[1])
                    break
            keys.remove(tup[1])
        mat[tup[0], tup[1]] = 0

#create subsetted dataframes, drop rows of each dataframe that have rows with NaN's, and store in 'data_dict'
#make dictionary of how many columns a row has per subsetted dataframe
row_count = {key:[] for key in df.index.values}
data_dict = {}
for group in groups:
    dfx = 'df{}'.format(groups.index(group)+1)
    col_idx = list(group)
    col_idx.append(df_l.shape[1]-1)
    dataframe = df_l.iloc[:,col_idx]
    dataframe.dropna(how='any', inplace=True)
    temp = np.zeros(dataframe.shape[0])
    temp.fill(len(col_idx)-1)
    count_full = np.array([dataframe.index.values, temp]).T
    data_dict[dfx] = dataframe
    second_counter = 0
    for entry in count_full:
        if second_counter % 10000 == 0:
            print second_counter
        row_count[int(entry[0])].append((int(entry[1]), dfx))
        second_counter += 1
    counter += 1

#make dictionary of what dataframe has the highest column count for every row
keepers = {}
empties = set()
for row, lst in row_count.iteritems():
    if len(lst) == 0:
        empties.add(row)
        continue
    lst.sort()
    keepers[row] = [tup[1] for tup in lst[-len(set(np.where(np.array(lst) == max(lst))[0])):]]

#make a dataframe of rows with all NaNs
df_l.loc[list(empties), 'Response'].to_csv('data/sub_frames/nulls.csv')

#transpose 'keepers' to associate a single dataframe with a unique list of rows
keep = {'df{}'.format(x):[] for x in xrange(1, len(data_dict)+1)}
for row, lst in keepers.iteritems():
    for dfn in lst:
        keep[dfn].append(row)

#subset dataframes and save as .csv
counter = 0
for name, df_ in data_dict.iteritems():
    new = df_.loc[keep[name]]
    if len(new)>0:
        counter += 1
        new.to_csv('data/sub_frames/df{}'.format(counter)+'.csv')
