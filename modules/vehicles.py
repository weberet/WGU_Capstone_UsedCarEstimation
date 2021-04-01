#CSV reading stuff
import csv
import os
#pandas
import pandas
pandas.set_option('display.expand_frame_repr', False)
#numpy
import numpy
#ML scikitlearn library
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale, StandardScaler
#Datetime
from datetime import date

def main():
    #Data file csv info
    filedirectory =  os.getcwd()
    filename= '../vehicleDB.csv'
    datafile = filedirectory+os.sep+filename

    #Old Code to read via chunks due to large dataset:
    #vdata = pandas.DataFrame()
    #for chunk in pandas.read_csv(datafile,chunksize=1000,error_bad_lines=False,engine='python',usecols=['price','year','manufacturer','model','condition','odometer']):
    #    vdata = pandas.concat([vdata, chunk], ignore_index=True)

    #New code using only optimized columns
    vdata = pandas.read_csv(datafile,usecols=['price','year','manufacturer','model','condition','odometer','title_status', 'cylinders','fuel'])

    print(vdata.dtypes)

    ################# Data-encoding ##################
    #vdata['manufacturer'] = vdata['manufacturer'].astype('category')
    #vdata['manufacturer'] = LabelEncoder().fit_transform(vdata['manufacturer'])

    # vdata = onehot_encode(vdata,['model'],['model'])
    vdata['model'] = vdata['model'].astype('category')
    vdata['model'] = LabelEncoder().fit_transform(vdata['model'])

    vdata['condition'] = vdata['condition'].astype('category')
    vdata['condition'] = LabelEncoder().fit_transform(vdata['condition'])

    #Changing feature of year into "age" for better model
    vdata['age'] = [date.today().year - i for i in vdata['year'].tolist()]
    vdata.drop('year', axis=1, inplace = True)
    ############## End Data-encoding #################

    ################# Data-cleaning ##################

    #Drop NaN values
    vdata = vdata.dropna()

    #Keep only clean or better cars to remove outliers
    #vdata = vdata[((vdata['title_status'] != 'missing') & (vdata['title_status'] != 'parts only')
    #    & (vdata['title_status'] != 'salvage') & (vdata['title_status'] != 'rebuilt')
    #   & (vdata['title_status'] != 'lien'))]

    #Condition to "rating":
    #vdata['condition'].replace('new', 6, inplace = True)
    #vdata['condition'].replace('like new', 5, inplace = True)
    #vdata['condition'].replace('excellent', 4, inplace = True)

    #remove Some Outliers pricewise:
    vdata['price'] = vdata[vdata['price'] < 99999]
    vdata['price'] = vdata[vdata['price'] > 800]

    #Outliers yearwise removal
    vdata = vdata[vdata['age'] < 25]
    vdata = vdata[vdata['age'] > 5]

    #Outliers odometer removal
    vdata = vdata[vdata['odometer'] < 225000]
    vdata = vdata[vdata['odometer'] > 100000]
    #vdata[['odometer']] = minmax_scale(vdata[['odometer']])

    #Cylinders to ints
    vdata['cylinders'] = vdata['cylinders'].replace('other', numpy.NaN)
    vdata['cylinders'] = vdata['cylinders'].str.extract('(\d+)', expand=False)
    #Drop NaN values
    vdata = vdata.dropna()

    vdata['cylinders'] = vdata['cylinders'].astype(int)
    print('Got here again')

    #vdata['cylinders'] = vdata[vdata['cylinders'] > 3]
    #vdata['cylinders'] = vdata[vdata['cylinders'] < 9]

    ##Onehot encode necessary stuff
    vdata = onehot_encode(vdata,['title_status','manufacturer','fuel'],['title','manu','fuel'])

    ################# End Data-cleaning ##################

    print(vdata.dtypes)
    print(vdata)

    ################# Training ##################
    #Seperate label and features into different dataframes
    #X = vdata[['age','manufacturer','condition','odometer', 'cylinders']]
    X = vdata.drop(['price'], axis=1)
    y = vdata[['price']]

    ################# Data Scaling ##################
    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    ############### End Data Scaling ################

    #Split data into test verification set and training set
    X_Train, X_Test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=1)
    print('got here')

    print('Training:\n')
    mlModel = LinearRegression() #create model object
    mlModel.fit(X_Train,y_train) #train model object

    #Validation
    print('Prediction:\n')
    prediction = mlModel.predict(X_Test)
    print(prediction)

    #Determine mean absoloute error
    meanError = mean_absolute_error(y_test,prediction)
    print(f'Mean Absoloute Error = {meanError}')
    #Root Mean Squared Error:
    rMeanSquareE = mean_squared_error(y_test,prediction)
    rMeanSquareE = numpy.sqrt(rMeanSquareE)
    print(f'Mean Square Error = {rMeanSquareE}')

    r2score = r2_score(y_test,prediction)
    print(f'r2score = {r2score}')

    print(f'Lin Regression R2 Score 2: {mlModel.score(X_Test,y_test)}')

def onehot_encode(df, columns, prefixes):
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pandas.get_dummies(df[column], prefix=prefix)
        df = pandas.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

if __name__ == "__main__":
    main()
    pass
else:
    print('Called as external module')