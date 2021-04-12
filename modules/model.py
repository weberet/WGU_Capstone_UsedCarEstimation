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
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale, StandardScaler
#Datetime
from datetime import date

#TODO future: make get_dataset(), make_light_csv() and get_dataset_forEDA() all the same method with argument toggling between each ones behaviors.
#Right now these three methods are essentially the same with different end results.

#Whether to use full or light version of DB (Original DB too big for github and heroku)
#To use the light version you must have the original turned into the light version using the make_light_csv() function
#to create a local copy
uselightDB = False;
originalVDBName = 'vehicleDB.csv'
lightVDBName = 'vehicleDBLight.csv'

def get_dataset():
    #Data file csv info
    filedirectory =  os.getcwd()
    #Original dataset CSV comes from Kaggle
    #https://www.kaggle.com/austinreese/craigslist-carstrucks-data
    if not uselightDB:
        filename = originalVDBName
    else:
        filename = lightVDBName
    datafile = filedirectory+os.sep+filename

    #New code using only optimized columns
    vdata = pandas.read_csv(datafile,usecols=['price','year','manufacturer','type','condition','odometer','title_status', 'cylinders','fuel'])
    #Columns used: price, age, manufacturer, model, condition, odometer, title_status, cylinders, fuel

    print(vdata.dtypes)

    ################# Data-encoding ##################
    #vdata['manufacturer'] = vdata['manufacturer'].astype('category')
    #vdata['manufacturer'] = LabelEncoder().fit_transform(vdata['manufacturer'])

    # vdata = onehot_encode(vdata,['model'],['model'])
    #vdata['model'] = vdata['model'].astype('category')
    #vdata['model'] = LabelEncoder().fit_transform(vdata['model'])

    vdata['condition'] = vdata['condition'].astype('category')
    #vdata['condition'] = LabelEncoder().fit_transform(vdata['condition'])

    #Changing feature of year into "age" for better model
    vdata['age'] = [date.today().year - i for i in vdata['year'].tolist()]
    vdata.drop('year', axis=1, inplace = True)
    ############## End Data-encoding #################

    ################# Data-cleaning ##################

    #Drop NaN values
    vdata = vdata.dropna()

    #Keep only clean or better cars to remove outliers
    vdata = vdata[((vdata['title_status'] != 'missing') & (vdata['title_status'] != 'parts only') & (vdata['title_status'] != 'salvage'))]
    # & (vdata['title_status'] != 'rebuilt') & (vdata['title_status'] != 'lien')

    #Condition to "rating":
    vdata['condition'].replace('new', 6, inplace = True)
    vdata['condition'].replace('like new', 5, inplace = True)
    vdata['condition'].replace('excellent', 4, inplace = True)
    vdata['condition'].replace('good', 3, inplace = True)
    vdata['condition'].replace('fair', 2, inplace = True)
    vdata['condition'].replace('salvage', 1, inplace = True)

    #remove Some Outliers pricewise:
    vdata['price'] = vdata[vdata['price'] < 99999]
    vdata['price'] = vdata[vdata['price'] > 800]

    #Outliers yearwise removal
    vdata = vdata[vdata['age'] < 25]
    #vdata = vdata[vdata['age'] > 5]

    #Outliers odometer removal
    vdata = vdata[vdata['odometer'] < 225000]
    vdata = vdata[vdata['odometer'] > 100000]
    #vdata[['odometer']] = minmax_scale(vdata[['odometer']])

    #Cylinders to ints
    vdata['cylinders'] = vdata['cylinders'].replace('other', numpy.NaN)
    if not uselightDB:
        vdata['cylinders'] = vdata['cylinders'].str.extract('(\d+)', expand=False)

    #Drop NaN values
    vdata = vdata.dropna()

    vdata['cylinders'] = vdata['cylinders'].astype(int)
    #vdata = vdata[vdata.cylinders != 3]
    vdata = vdata[vdata.cylinders != 10]
    vdata = vdata[vdata.cylinders != 12]

    #Drop unfit / exotic brands:
    #vdata = vdata[vdata.manufacturer != 'harley-davidson']

    ##Onehot encode necessary stuff
    vdata = onehot_encode(vdata,['title_status','manufacturer','fuel','type'],['title','manu','fuel','type'],False)

    ################# End Data-cleaning ##################

    print(vdata.dtypes)
    print(vdata)
    return vdata

#Creates a light version of database for use on github / online repositories.
#Copied originally from the get_dataset_forEDA() function as this is closer to original dataset.
#Download original dataset for more extensive options
#Original Dataset:
#Original dataset CSV comes from Kaggle
#https://www.kaggle.com/austinreese/craigslist-carstrucks-data
def make_light_csv():
    # Data file csv info
    filedirectory = os.getcwd()
    filename = originalVDBName
    datafile = filedirectory + os.sep + filename

    # New code using only optimized columns
    vdata = pandas.read_csv(datafile,
                            usecols=['price', 'year', 'manufacturer', 'type', 'condition', 'odometer', 'title_status',
                                     'cylinders', 'fuel'])
    # Columns used: price, age, manufacturer, model, condition, odometer, title_status, cylinders, fuel

    print(vdata.dtypes)

    ################# Data-encoding ##################
    # vdata['manufacturer'] = vdata['manufacturer'].astype('category')
    # vdata['manufacturer'] = LabelEncoder().fit_transform(vdata['manufacturer'])

    # vdata = onehot_encode(vdata,['model'],['model'])
    # vdata['model'] = vdata['model'].astype('category')
    # vdata['model'] = LabelEncoder().fit_transform(vdata['model'])

    vdata['condition'] = vdata['condition'].astype('category')
    # vdata['condition'] = LabelEncoder().fit_transform(vdata['condition'])

    # Changing feature of year into "age" for better model
    vdata['age'] = [date.today().year - i for i in vdata['year'].tolist()]
    ############## End Data-encoding #################

    ################# Data-cleaning ##################

    # Drop NaN values
    vdata = vdata.dropna()

    # Keep only clean or better cars to remove outliers
    vdata = vdata[((vdata['title_status'] != 'missing') & (vdata['title_status'] != 'parts only') & (
                vdata['title_status'] != 'salvage'))]
    # & (vdata['title_status'] != 'rebuilt') & (vdata['title_status'] != 'lien')

    # Condition to "rating":
    vdata['condition'].replace('new', 6, inplace=True)
    vdata['condition'].replace('like new', 5, inplace=True)
    vdata['condition'].replace('excellent', 4, inplace=True)
    vdata['condition'].replace('good', 3, inplace=True)
    vdata['condition'].replace('fair', 2, inplace=True)
    vdata['condition'].replace('salvage', 1, inplace=True)

    # remove Some Outliers pricewise:
    vdata['price'] = vdata[vdata['price'] < 99999]
    vdata['price'] = vdata[vdata['price'] > 800]

    # Outliers yearwise removal
    vdata = vdata[vdata['age'] < 25]
    #vdata = vdata[vdata['age'] > 5]

    # Outliers odometer removal
    vdata = vdata[vdata['odometer'] < 225000]
    vdata = vdata[vdata['odometer'] > 100000]
    # vdata[['odometer']] = minmax_scale(vdata[['odometer']])

    # Cylinders to ints
    vdata['cylinders'] = vdata['cylinders'].replace('other', numpy.NaN)
    vdata['cylinders'] = vdata['cylinders'].str.extract('(\d+)', expand=False)

    # Drop NaN values
    vdata = vdata.dropna()

    vdata['cylinders'] = vdata['cylinders'].astype(int)
    # vdata = vdata[vdata.cylinders != 3]
    vdata = vdata[vdata.cylinders != 10]
    vdata = vdata[vdata.cylinders != 12]

    # Drop unfit / exotic brands:
    # vdata = vdata[vdata.manufacturer != 'harley-davidson']

    ################# End Data-cleaning ##################

    vdata.to_csv(lightVDBName,index=False)

#Gets dataset for Exploratory Data Analysis from SweetViz (No one hot encoding, years left intact instead of age)
def get_dataset_forEDA():
    #Data file csv info
    filedirectory =  os.getcwd()
    if not uselightDB:
        filename = originalVDBName
    else:
        filename = lightVDBName
    datafile = filedirectory+os.sep+filename

    #New code using only optimized columns
    vdata = pandas.read_csv(datafile,usecols=['price','year','manufacturer', 'type','condition','odometer','title_status', 'cylinders','fuel'])
    #Columns used: price, age, manufacturer, model, condition, odometer, title_status, cylinders, fuel

    print(vdata.dtypes)

    ################# Data-encoding ##################
    #vdata['manufacturer'] = vdata['manufacturer'].astype('category')
    #vdata['manufacturer'] = LabelEncoder().fit_transform(vdata['manufacturer'])

    # vdata = onehot_encode(vdata,['model'],['model'])
    #vdata['model'] = vdata['model'].astype('category')
    #vdata['model'] = LabelEncoder().fit_transform(vdata['model'])

    vdata['condition'] = vdata['condition'].astype('category')
    #vdata['condition'] = LabelEncoder().fit_transform(vdata['condition'])

    #Changing feature of year into "age" for better model
    vdata['age'] = [date.today().year - i for i in vdata['year'].tolist()]
    ############## End Data-encoding #################

    ################# Data-cleaning ##################

    #Drop NaN values
    vdata = vdata.dropna()

    #Keep only clean or better cars to remove outliers
    vdata = vdata[((vdata['title_status'] != 'missing') & (vdata['title_status'] != 'parts only') & (vdata['title_status'] != 'salvage'))]
    # & (vdata['title_status'] != 'rebuilt') & (vdata['title_status'] != 'lien')

    #Condition to "rating":
    vdata['condition'].replace('new', 6, inplace = True)
    vdata['condition'].replace('like new', 5, inplace = True)
    vdata['condition'].replace('excellent', 4, inplace = True)
    vdata['condition'].replace('good', 3, inplace = True)
    vdata['condition'].replace('fair', 2, inplace = True)
    vdata['condition'].replace('salvage', 1, inplace = True)

    #remove Some Outliers pricewise:
    vdata['price'] = vdata[vdata['price'] < 99999]
    vdata['price'] = vdata[vdata['price'] > 800]

    #Outliers yearwise removal
    vdata = vdata[vdata['age'] < 25]
    #vdata = vdata[vdata['age'] > 5]

    #Outliers odometer removal
    vdata = vdata[vdata['odometer'] < 225000]
    vdata = vdata[vdata['odometer'] > 100000]
    #vdata[['odometer']] = minmax_scale(vdata[['odometer']])

    #Cylinders to ints
    vdata['cylinders'] = vdata['cylinders'].replace('other', numpy.NaN)
    if not uselightDB:
        vdata['cylinders'] = vdata['cylinders'].str.extract('(\d+)', expand=False)

    #Drop NaN values
    vdata = vdata.dropna()

    vdata['cylinders'] = vdata['cylinders'].astype(int)
    #vdata = vdata[vdata.cylinders != 3]
    vdata = vdata[vdata.cylinders != 10]
    vdata = vdata[vdata.cylinders != 12]

    #Drop unfit / exotic brands:
    #vdata = vdata[vdata.manufacturer != 'harley-davidson']

    ################# End Data-cleaning ##################

    print(vdata.dtypes)
    print(vdata)
    return vdata

def get_X(dataset):
    X = dataset.drop(['price','title_status','manufacturer','fuel', 'type'], axis=1) #Skip these columns as not used or using onehot encoding.

    scaler = get_StandardScaler(dataset)
    print(f'X = {X}')

    X = scaler.fit_transform(X)
    return X

def get_StandardScaler(dataset):
    print('printing dataset for standard scaler')
    print(dataset)
    print(f'Pre Split shape: {dataset.shape}')
    X = dataset.drop(['price','title_status','manufacturer','fuel','type'], axis=1) #Skip these columns as not used or using onehot encoding.
    print(f'X shape: {X.shape}')
    print(f'Dataset after: {X}')
    scaler = StandardScaler()

    scaler.fit_transform(X)
    return scaler

def get_y(dataset):
    y = dataset[['price']] #column predictions target
    return y

def get_trained_model(X,y):
    #Split data into test verification set and training set
    X_Train, X_Test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=1)
    print('got here')

    print('Training:\n')
    #Switching to Poisson from Linear brought RMSE down from 2614.92 to 2281.12
    mlModel = PoissonRegressor() #create model object #Switched from LinearRegression to PoissonRegressor
    mlModel.fit(X_Train,y_train.values.ravel()) #train model object
    return mlModel

def get_train_split_var(X,y,which):
    X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    if(which == 'X_Train'):
        return X_Train
    elif(which == 'X_Test'):
        return X_Test
    elif(which == 'y_train'):
        return y_train
    elif(which == 'y_test'):
        return y_test
    else:
        print('Unrecognized which parameter, returning None')
        return None

def get_r2(model,X_Test,y_test):
    prediction = model.predict(X_Test)

    r2score = r2_score(y_test,prediction)
    #print(f'r2score = {r2score}')
    return r2score

def get_mean_error(model,X_Test,y_test):
    prediction = model.predict(X_Test)

    #Determine mean absoloute error
    meanError = mean_absolute_error(y_test,prediction)
    #print(f'Mean Absoloute Error = {meanError}')
    return meanError

def get_rmse(model, X_Test,y_test):
    prediction = model.predict(X_Test)
    #Root Mean Squared Error:
    rMeanSquareE = mean_squared_error(y_test,prediction)
    rMeanSquareE = numpy.sqrt(rMeanSquareE)
    #print(f'Mean Square Error = {rMeanSquareE}')
    return rMeanSquareE

def get_unique_values(dataset,column):
    return dataset[column].unique()

def onehot_encode(df, columns, prefixes,drop):
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pandas.get_dummies(df[column], prefix=prefix)
        df = pandas.concat([df, dummies], axis=1)
        if(drop == True):
            df = df.drop(column, axis=1)
    return df

if __name__ == "__main__":
    print(f'Called as main module')
    pass
else:
    print(f'{__name__} called as external module.')