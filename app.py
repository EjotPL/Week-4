from flask import Flask
import joblib
from flask import jsonify
import pandas as pd
# Create Flask app
app = Flask(__name__)

model = joblib.load("C:\\Users\\ejot9\\linear_regression.joblib")


@app.route('/predict', methods=['GET'])
def index():
    data = pd.read_csv("C:\\Users\\ejot9\\toy_data.csv")
    # Extract the feature data as a 2D array
    feature_data = data.values[:, :-1]
    predicted_data = model.predict(feature_data)
    return jsonify({'predicted_data': predicted_data.tolist()})
 

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




