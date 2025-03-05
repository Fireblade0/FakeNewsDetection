from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request (AJAX request sends JSON)
    data = request.get_json()
    news_title = data['title']
    
    # Vectorize the input text
    news_title_vectorized = vectorizer.transform([news_title])
    
    # Make prediction
    prediction = model.predict(news_title_vectorized)[0]
    
    # Convert prediction to a regular Python integer (to avoid numpy int64 error)
    prediction = int(prediction)
    
    # Return the result as JSON
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
