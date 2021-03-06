from datetime import date
import modules.model as myModel
import modules.modifyReport as modReport
import pandas
import locale
import sweetviz as sv
import os
import csv
import json
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, make_response
from flask import templating
import pickle

app = Flask(__name__)

herokuStartupEnabled = False  # Variable that makes app more compatible with heroku free tier

useSavedModel = False
modelfilename = 'trained_model.sav'

# Set locale:
locale.setlocale(locale.LC_ALL, '')

if not useSavedModel:
    # region Model Train
    # Global model creation for start of application
    data = myModel.get_dataset()
    X = myModel.get_X(data)
    y = myModel.get_y(data)
    trainedModel = myModel.get_trained_model(X, y)
    meanAbsoluteError = myModel.get_mean_error(trainedModel, myModel.get_train_split_var(X, y, 'X_Test'),
                                           myModel.get_train_split_var(X, y, 'y_test'))
    rootMeanSquaredError = myModel.get_rmse(trainedModel, myModel.get_train_split_var(X, y, 'X_Test'),
                                           myModel.get_train_split_var(X, y, 'y_test'))

    r2 = myModel.get_r2(trainedModel, myModel.get_train_split_var(X, y, 'X_Test'),
                                           myModel.get_train_split_var(X, y, 'y_test'))
    standardScaler = myModel.get_StandardScaler(data)

    # singlePrediction = trainedModel.predict(['acura',4,8,].reshape(1,-1))
    # print(f'Single prediction {singlePrediction}')
    print(f'Mean Absolute  for trained model = {meanAbsoluteError}')
    print(f'Root Mean Squared Error for trained model = {rootMeanSquaredError}')
    print(f'R2 for trained model = {r2}')
    print(f'Saving ML model...')
    pickle.dump(trainedModel, open(modelfilename, 'wb+'))
    # endregion
else:
    trainedModel = pickle.load(open(modelfilename, 'rb'))

# region Menu Options
manufacturerOptions = sorted(myModel.get_unique_values(data, 'manufacturer').tolist())
print(f'Manufacturer options:\n {manufacturerOptions}')

typeOptions = sorted(myModel.get_unique_values(data, 'type').tolist())
print(f'Type options:\n {typeOptions}')

# yearOptions population
yearOptions = []
today = date.today()
thisyear = today.year
for i in range(0, 25):
    yearOptions.append(thisyear - i)

# conditionOptions = myModel.get_unique_values(data,'condition').tolist()
conditionOptions = ['salvage', 'fair', 'good', 'excellent', 'like new', 'new']

titleOptions = myModel.get_unique_values(data, 'title_status').tolist()
print(titleOptions)
# cylinderOptions = myModel.get_unique_values(data,'cylinders').tolist()
print(f'Unique cylinder options = {myModel.get_unique_values(data, "cylinders").tolist()}')
cylinderOptions = [3, 4, 5, 6, 8]
fuelOptions = myModel.get_unique_values(data, 'fuel').tolist()
# endregion

estimateDBFile = 'estimateDB.csv'


# Run below line while original CSV is available to create a light version for upload to heroku / github
if not myModel.uselightDB: #Makes light version of db for future use if original is being used. Mainly for heroku / github
    myModel.make_light_csv()

@app.route('/')
def index():
    return render_template('index.html', manufacturers=manufacturerOptions, type=typeOptions, years=yearOptions,
                           conditions=conditionOptions,
                           titles=titleOptions, cylinders=cylinderOptions, fuels=fuelOptions)


@app.route('/predict', methods=['POST'])
def predict():
    recievedJSON = request.get_json()
    print(f'recieved Data = {recievedJSON}')

    estimation = makePrediction(recievedJSON['manufacturer'], recievedJSON['type'], int(recievedJSON['year']),
                                recievedJSON['condition'], recievedJSON['title'], recievedJSON['cylinder'],
                                recievedJSON['fuel'], int(recievedJSON['mileage']))

    returnJson = {'value': str(estimation)}

    response = make_response(returnJson, 200)
    response.mimetype = "application/json"

    print(f'Response = {response}')

    # Entry into csv database
    estimationForDB = str(estimation).replace('"', '').replace('$', '').replace(',', '')
    estimateEntry(estimateDBFile, manufacturer=recievedJSON['manufacturer'], vtype=recievedJSON['type'],
                  year=int(recievedJSON['year']),
                  condition=recievedJSON['condition'], title=recievedJSON['title'], cylinders=recievedJSON['cylinder'],
                  fuel=recievedJSON['fuel'], mileage=int(recievedJSON['mileage']), estimate=estimationForDB)

    return response
    #
    # return returnData, 200, {'Content-Type': 'text/xml; charset=utf-8'}
    # makePrediction('acura',2011,'excellent','rebuilt',4,'diesel',120000)


@app.route('/graphs')
def graphs():
    return 'Graphs page'


@app.route('/stats')
def stats():
    # Read the estimation CSV Database to pandas dataframe
    estimateDF = pandas.read_csv(estimateDBFile)
    n = 1  # Get only top result guessed, adjust this to get more results for lines using n

    # variable assignment
    statnumPredictions = len(estimateDF.index)
    statmanufacturer = estimateDF["manufacturer"].value_counts()[:n].index.tolist()[0]
    stattype = estimateDF["type"].value_counts()[:n].index.tolist()[0]
    statyear = estimateDF["year"].mean()
    statcondition = estimateDF["condition"].value_counts()[:n].index.tolist()[0]
    stattitle = estimateDF["title"].value_counts()[:n].index.tolist()[0]
    statcylinders = estimateDF["cylinders"].value_counts()[:n].index.tolist()[0]
    statfuel = estimateDF["fuel"].value_counts()[:n].index.tolist()[0]
    statmileage = estimateDF["mileage"].mean()
    statestimation = estimateDF["estimate"].mean()

    print(f'Number of estimations calculated with this tool: {statnumPredictions}')
    print(f'Top guessed vehicle manufacturer: {statmanufacturer}')
    print(f'Top guessed vehicle type: {stattype}')
    print(f'Average year of estimated vehicles: {statyear}')
    print(f'Most common condition of user vehicle: {statcondition}')
    print(f'Most common title of user vehicle: {stattitle}')
    print(f'Most common number of cylinders in user vehicle: {statcylinders}')
    print(f'Most common fuel type among estimated vehicles: {statfuel}')
    print(f'Average mileage of estimated vehicles: {statmileage}')
    print(f'Average price estimation of vehicles: {statestimation}')

    return render_template('stats.html', statmae=meanAbsoluteError, statrmse=rootMeanSquaredError, statr2=r2,
                           statnumPredictions=statnumPredictions, statmanufacturer=statmanufacturer,
                           stattype=stattype, statyear=statyear, statcondition=statcondition, stattitle=stattitle,
                           statcylinders=statcylinders, statfuel=statfuel, statmileage=statmileage,
                           statestimation=statestimation)


@app.route('/genreport')  # Generate report code
def genreport():
    # region SweetViz Stuff
    sv.config_parser.read('svOverride.ini')  # SweetViz Overrides
    dfReport = sv.analyze(myModel.get_dataset_forEDA())
    filedirectory = os.getcwd()
    templatesString = 'templates'
    filename = 'svreport.html'
    dirString = filedirectory + os.sep + templatesString + os.sep + filename
    dfReport.show_html(filepath=dirString, open_browser="False", layout='widescreen')
    modReport.processSaveReport()  # Custom modifications to report
    return ('', 204)
    # endregion


@app.route('/edagraphs')  # Generate report code
def edagraphs():
    # region SweetViz Stuff
    return render_template('svreport.html')
    # endregion


def makePrediction(manufacturer, vtype, year, condition, title, cylinder, fuel, mileage):
    predDictionary = {
        'condition': [0],
        'cylinders': [0],
        'odometer': [0],
        'age': [0],
        'title_clean': [0],
        'title_lien': [0],
        'title_rebuilt': [0],
        'manu_acura': [0],
        'manu_audi': [0],
        'manu_bmw': [0],
        'manu_buick': [0],
        'manu_cadillac': [0],
        'manu_chevrolet': [0],
        'manu_chrysler': [0],
        'manu_dodge': [0],
        'manu_fiat': [0],
        'manu_ford': [0],
        'manu_gmc': [0],
        'manu_harley-davidson': [0],
        'manu_honda': [0],
        'manu_hyundai': [0],
        'manu_infiniti': [0],
        'manu_jaguar': [0],
        'manu_jeep': [0],
        'manu_kia': [0],
        'manu_land rover': [0],
        'manu_lexus': [0],
        'manu_lincoln': [0],
        'manu_mazda': [0],
        'manu_mercedes-benz': [0],
        'manu_mercury': [0],
        'manu_mini': [0],
        'manu_mitsubishi': [0],
        'manu_nissan': [0],
        'manu_pontiac': [0],
        'manu_porsche': [0],
        'manu_ram': [0],
        'manu_rover': [0],
        'manu_saturn': [0],
        'manu_subaru': [0],
        'manu_toyota': [0],
        'manu_volkswagen': [0],
        'manu_volvo': [0],
        'fuel_diesel': [0],
        'fuel_electric': [0],
        'fuel_gas': [0],
        'fuel_hybrid': [0],
        'fuel_other': [0],
        'type_SUV': [0],
        'type_bus': [0],
        'type_convertible': [0],
        'type_coupe': [0],
        'type_hatchback': [0],
        'type_mini-van': [0],
        'type_offroad': [0],
        'type_other': [0],
        'type_pickup': [0],
        'type_sedan': [0],
        'type_truck': [0],
        'type_van': [0],
        'type_wagon': [0]
    }

    # Create single entry dataframe
    predDF = pandas.DataFrame.from_dict(predDictionary)

    # Modify dataframe depending on function input:
    predDF.at[0, 'manu_' + manufacturer] = 1
    predDF.at[0, 'type_' + vtype] = 1
    predDF.at[0, 'age'] = thisyear - year
    predDF.at[0, 'condition'] = mapCondition(condition)
    predDF.at[0, 'title_' + title] = 1
    predDF.at[0, 'cylinders'] = cylinder
    predDF.at[0, 'fuel_' + fuel] = 1
    predDF.at[0, 'odometer'] = mileage

    # Make prediction
    print(f'Prediction DataFrame = \n{predDF}')
    print(f'StandardScaler Required Fields: {standardScaler.get_params()}')
    aPred = trainedModel.predict(standardScaler.transform(predDF))
    print(f'A Prediction = {aPred[0]}')  # If standard linear regression index is aPred[0][0]
    returnString = locale.currency(round(aPred[0], 2),
                                   grouping=True)  # If standard linear regression index is aPred[0][0]
    return returnString


def mapCondition(condition):
    if (condition == 'new'):
        return 6
    elif (condition == 'like new'):
        return 5
    elif (condition == 'excellent'):
        return 4
    elif (condition == 'good'):
        return 3
    elif (condition == 'fair'):
        return 2
    elif (condition == 'salvage'):
        return 1
    else:
        print('Unrecognized condition, returning None')
        return None


def checkDBExist(DBfile):
    if not os.path.isfile(DBfile):
        print("DBfile CSV doesn't exist, creating")
        with open(DBfile, 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            filewriter.writerow(
                ['manufacturer', 'year', 'condition', 'title', 'cylinders', 'fuel', 'mileage', 'estimate'])
            csvfile.close()


def estimateEntry(DBfile, manufacturer, vtype, year, condition, title, cylinders, fuel, mileage, estimate):
    checkDBExist(DBfile)
    with open(DBfile, 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        filewriter.writerow([manufacturer, vtype, year, condition, title, cylinders, fuel, mileage, estimate])
        csvfile.close()


def dummyPredict(DBfile, man, yea, con, tit, cyl, fue,
                 mil):  # Dummy prediction method allowing for creation of dummy entries to prevent db not existing with heroku free tier.
    estimation = makePrediction(man, yea, con, tit, cyl, fue, mil)

    # Entry into csv database
    estimationForDB = str(estimation).replace('"', '').replace('$', '').replace(',', '')
    estimateEntry(DBfile, man, int(yea), con, tit, cyl, fue, int(mil), estimationForDB)
    print('Made dummy entry')


def dummyGenReport():  # Method for initial creation of reports on platforms like heroku
    # region SweetViz Stuff
    sv.config_parser.read('svOverride.ini')  # SweetViz Overrides
    dfReport = sv.analyze(myModel.get_dataset_forEDA())
    filedirectory = os.getcwd()
    templatesString = 'templates'
    filename = 'svreport.html'
    dirString = filedirectory + os.sep + templatesString + os.sep + filename
    dfReport.show_html(filepath=dirString, open_browser="False", layout='widescreen')
    modReport.processSaveReport()  # Custom modifications to report
    # endregion


if herokuStartupEnabled:
    dummyPredict(estimateDBFile, 'toyota', 1998, 'fair', 'clean', 4, 'gas', 180000)
    # dummyGenReport() #Create EDA reports so that the exist on heroku free tier without clicking regenerate
    # dummy reports take to long to generate with heroku timeout request, serve up pregenerated with option to regen.

if __name__ == '__main__':
    app.run(debug=True, use_debugger=False, use_reloader=False, passthrough_errors=True)
