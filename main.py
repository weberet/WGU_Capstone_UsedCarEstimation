from datetime import date
import modules.model as myModel
import modules.modifyReport as modReport
import pandas
import locale
import sweetviz as sv
import os
import json
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, make_response
from flask import templating
app = Flask(__name__)

#region App Startup
#Set locale:
locale.setlocale( locale.LC_ALL, '' )

#Global model creation for start of application
data = myModel.get_dataset()
#region Menu Options
manufacturerOptions = sorted(myModel.get_unique_values(data,'manufacturer').tolist())
print(f'Manufacturer options:\n {manufacturerOptions}')

#yearOptions population
yearOptions = []
today = date.today()
thisyear = today.year
for i in range(5,25):
    yearOptions.append(thisyear - i)

#conditionOptions = myModel.get_unique_values(data,'condition').tolist()
conditionOptions = ['salvage','fair','good','excellent','like new', 'new']

titleOptions = myModel.get_unique_values(data,'title_status').tolist()
print(titleOptions)
#cylinderOptions = myModel.get_unique_values(data,'cylinders').tolist()
print(f'Unique cylinder options = {myModel.get_unique_values(data,"cylinders").tolist()}')
cylinderOptions = [3,4,5,6,8]
fuelOptions = myModel.get_unique_values(data,'fuel').tolist()
#endregion
X = myModel.get_X(data)
y = myModel.get_y(data)
trainedModel = myModel.get_trained_model(X,y)
rootMeanError = myModel.get_mean_error(trainedModel,myModel.get_train_split_var(X,y,'X_Test'),
                             myModel.get_train_split_var(X,y,'y_test'))
standardScaler = myModel.get_StandardScaler(data)

#singlePrediction = trainedModel.predict(['acura',4,8,].reshape(1,-1))
#print(f'Single prediction {singlePrediction}')
print(f'Root Mean Error for trained model = {rootMeanError}')
#endregion

@app.route('/')
def index():
    return render_template('index.html',manufacturers=manufacturerOptions,years=yearOptions,conditions=conditionOptions,
                           titles=titleOptions,cylinders=cylinderOptions,fuels=fuelOptions)

@app.route('/predict', methods=['POST'])
def predict():

    recievedJSON = request.get_json()
    print(f'recieved Data = {recievedJSON}')

    estimation = makePrediction(recievedJSON['manufacturer'],int(recievedJSON['year']),recievedJSON['condition'],
                                recievedJSON['title'],recievedJSON['cylinder'],recievedJSON['fuel'],
                                int(recievedJSON['mileage']))

    returnJson = {'value':str(estimation)}

    response = make_response(returnJson, 200)
    response.mimetype = "application/json"

    print(f'Response = {response}')

    return response
    #
    #return returnData, 200, {'Content-Type': 'text/xml; charset=utf-8'}
    #makePrediction('acura',2011,'excellent','rebuilt',4,'diesel',120000)

@app.route('/graphs')
def graphs():
    return 'Graphs page'

@app.route('/stats')
def stats():
    return  render_template('stats.html',numPredictions=5)

@app.route('/genreport') #Generate report code
def genreport():
    # region SweetViz Stuff
    sv.config_parser.read('svOverride.ini') #SweetViz Overrides
    dfReport = sv.analyze(myModel.get_dataset_forEDA())
    filedirectory = os.getcwd()
    templatesString = 'templates'
    filename = 'svreport.html'
    dirString = filedirectory + os.sep + templatesString + os.sep + filename
    dfReport.show_html(filepath=dirString, open_browser="False",layout='widescreen')
    modReport.processSaveReport() #Custom modifications to report
    return ('', 204)
    # endregion

@app.route('/edagraphs') #Generate report code
def edagraphs():
    # region SweetViz Stuff
    return render_template('svreport.html')
    # endregion


def makePrediction(manufacturer,year,condition,title,cylinder,fuel,mileage):
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
    }

    #Create single entry dataframe
    predDF = pandas.DataFrame.from_dict(predDictionary)

    #Modify dataframe depending on function input:
    predDF.at[0, 'manu_'+manufacturer] = 1
    predDF.at[0, 'age'] = thisyear - year
    predDF.at[0, 'condition'] = mapCondition(condition)
    predDF.at[0, 'title_'+title] = 1
    predDF.at[0, 'cylinders'] = cylinder
    predDF.at[0, 'fuel_'+fuel] = 1
    predDF.at[0, 'odometer'] = mileage
    print(f'predDataFrame = \n{predDF}')

    #Make prediction
    aPred = trainedModel.predict(standardScaler.transform(predDF))
    print(f'A Prediction = {aPred[0]}') #If standard linear regression index is aPred[0][0]
    returnString = locale.currency(round(aPred[0], 2),grouping=True) #If standard linear regression index is aPred[0][0]
    return returnString

def mapCondition(condition):

    if(condition == 'new'):
        return 6
    elif(condition == 'like new'):
        return 5
    elif(condition == 'excellent'):
        return 4
    elif(condition == 'good'):
        return 3
    elif(condition == 'fair'):
        return 2
    elif(condition == 'salvage'):
        return 1
    else:
        print('Unrecognized condition, returning None')
        return None



if __name__ == '__main__':
    app.run(debug=True, use_debugger=False, use_reloader=False, passthrough_errors=True)
