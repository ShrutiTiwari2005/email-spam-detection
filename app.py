from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data)
        prediction = model.predict(vect)
        result = "SPAM ❌" if prediction[0] else "NOT SPAM ✅"
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
