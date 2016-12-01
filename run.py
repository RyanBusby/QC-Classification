import cPickle as pickle
import pandas as pd

with open('data/valid_measures.pkl', 'rb') as f:
    vmsrs = pickle.load(f)

with open('data/col_dict.pkl', 'rb') as i:
    col_dict = pickle.load(i)

while True:
    line = raw_input('enter line (0-3): ')
    station = raw_input('enter station (0-51): ')
    feat = raw_input('enter feature (0-4263): ')
    measurement = 'L'+line+'_''S'+station+'_'+'F'+feat
    if measurement in vmsrs:
        break
    else:
        print 'Invalid, try again.'

for name, cols in col_dict.iteritems():
    if measurement in cols:
        cluster = name
        num = name[2:]
        break

if num == 14 or num == 9 or num == 19:
    pred = 0

else:
    df = pd.DataFrame(data=[range(len(col_dict[cluster]))], columns=col_dict[cluster])

    for col in col_dict[cluster]:
        df[col] = float(raw_input('enter '+col+': ' ))

    X = df.values

    with open('models/pca/pca'+num+'.pkl', 'rb') as p:
        pca = pickle.load(p)

    with open('models/models/mod'+num+'.pkl', 'rb') as m:
        mod = pickle.load(m)

    X = pca.transform(X)

    pred = mod.predict(X)

if pred == 1:
    print 'FAIL'

elif pred == 0:
    print 'PASS'
