from datetime import date
import modules.model as myModel
import pandas
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template
from flask import templating
app = Flask(__name__)

#Global model creation for start of application
data = myModel.get_dataset()
#region Menu Options
manufacturerOptions = sorted(myModel.get_unique_values(data,'manufacturer').tolist())
print(f'Manufacturer options:\n {manufacturerOptions}')

#yearOptions population
yearOptions = []
today = date.today()
for i in range(5,25):
    yearOptions.append(today.year - i)

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

@app.route('/')
def index():
    return render_template('index.html',manufacturers=manufacturerOptions,years=yearOptions,conditions=conditionOptions,
                           titles=titleOptions,cylinders=cylinderOptions,fuels=fuelOptions)
    #return 'Hello, World!'

@app.route('/predict')
def predict():
    return trainedModel.predict([vars])

@app.route('/graphs')
def graphs():
    return 'Graphs page'

@app.route('/stats')
def stats():
    return 'Stats page'

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

    predDF = pandas.DataFrame.from_dict(predDictionary)
    print(f'predDataFrame = \n{predDF}')
    aPred = trainedModel.predict(standardScaler.transform(predDF))
    print(f'A Prediction = {aPred[0][0]}')

makePrediction('acura',2016,'salvage','lien',4,'diesel',120000)

if __name__ == '__main__':
    app.run(debug=True, use_debugger=False, use_reloader=False, passthrough_errors=True)
