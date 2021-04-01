from datetime import date
import modules.model as myModel
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template
from flask import templating
app = Flask(__name__)

#Global model creation for start of application
data = myModel.get_dataset()
#region Menu Options
manufacturerOptions = myModel.get_unique_values(data,'manufacturer').tolist()
print(f'Manufacturer options:\n {manufacturerOptions}')

#yearOptions population
yearOptions = []
today = date.today()
for i in range(5,25):
    yearOptions.append(today.year - i)

conditionOptions = myModel.get_unique_values(data,'condition').tolist()
titleOptions = myModel.get_unique_values(data,'title_status').tolist()
cylinderOptions = myModel.get_unique_values(data,'cylinders').tolist()
fuelOptions = myModel.get_unique_values(data,'fuel').tolist()
#endregion
#X = myModel.get_X(data)
#y = myModel.get_y(data)
#trainedModel = myModel.get_trained_model(X,y)

@app.route('/')
def index():
    return render_template('index.html',manufacturers=manufacturerOptions,years=yearOptions,conditions=conditionOptions,
                           titles=titleOptions,cylinders=cylinderOptions,fuels=fuelOptions)
    #return 'Hello, World!'

@app.route('/predict')
def predict():
    return 'Prediction route'
    #return trainedModel.predict([vars])

@app.route('/graphs')
def graphs():
    return 'Graphs page'

@app.route('/stats')
def stats():
    return 'Stats page'



if __name__ == '__main__':
    app.run(debug=True, use_debugger=False, use_reloader=False, passthrough_errors=True)
