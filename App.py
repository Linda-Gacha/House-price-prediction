from flask import Flask, render_template,jsonify,request
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('HouseModel.pkl', 'rb'))
@app.route('/')
def hello_world():
   return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        sqft_living = int(request.form['sqft_living'])
        sqft_lot = int(request.form['sqft_lot'])
        floors = int(request.form['floors'])
        waterfront = int(request.form['waterfront'])
        condition = int(request.form['condition'])
        yr_built = int(request.form['yr_built'])
        input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, condition, yr_built]])
        predd = model.predict(input_data) 
        prediction  = predd.tolist()[0][0]*3
        formatted_pred = "{:.2f}".format(prediction)
        return render_template('index.html', prediction=formatted_pred)
    else:
        return render_template('index.html', prediction=0)
if __name__ == '__main__':
   app.debug = True
   app.run()

