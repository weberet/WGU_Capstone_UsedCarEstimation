This project is a WGU Capstone Project which uses a machine learning model along with a poisson regression algorithm to estimate used car prices based off of certain vehicle attributes. It's functionality is similar to that of the Kelly Blue Book price estimator in that it is attempting to guess a fair price value for any given vehicle. The application does this through training a model that examines many used vehicles in a dataset obtained from Craigslist.

The application is a Python Flask web app, which provides the web interface. It also incorprates libraries such as Scikit-learn and Sweetviz for machine learnign and exploratory data analysis.

A working copy of this applicaiton is hosted at https://acme-car-estimator.herokuapp.com/ be aware that, because this application is hosted using the Heroku free tier, the Heroku dyno on which it runs will likely be initially asleep, visiting this address for the first time may yield an error, however after waiting a minute or two and then refreshing the page the dyno will usually be ready to run the application and provide the user interface. 

This heroku version of the application uses a condensed version of the database that only contains necessary fields to avoid file upload limits. It still yields the same results as all the records contained in it are all the records it trains from in the original dataset.

Original Dataset is located and downloadable from:
https://www.kaggle.com/austinreese/craigslist-carstrucks-data
