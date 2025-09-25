from flask import Flask, render_template, request, jsonify
from model import predict_future, predict_until_date, load_data_and_model  # Import your model functions

app = Flask(__name__)

# Load data and model once at the start
data, scaler, model = load_data_and_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    days = int(request.json['days'])  # Dùng request.json thay vì request.form
    dates, rates = predict_future(days, model, scaler)
    return jsonify({'dates': [d.strftime('%Y-%m-%d') for d in dates], 'rates': rates})

@app.route('/predict_until', methods=['POST'])
def predict_until():
    target_date = request.form['target_date']
    target_date = pd.to_datetime(target_date, format="%Y-%m-%d")
    predictions = predict_until_date(target_date, model, scaler, data)
    return jsonify({'predictions': predictions})

if __name__ == "__main__":
    app.run(debug=True)
