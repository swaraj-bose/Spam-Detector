from flask import Flask, request, jsonify
import pickle
import os

# Load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict if a given message is spam or not.
    Accepts JSON data with a 'message' field.
    Returns JSON with the prediction result.
    """
    try:
        # Parse input data
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'Invalid input. Please provide a message.'}), 400

        # Extract message and make prediction
        message = data['message']
        vectorized_message = vectorizer.transform([message])
        prediction = model.predict(vectorized_message)[0]

        # Interpret the prediction
        result = 'Spam' if prediction == 1 else 'Not Spam'
        return jsonify({'result': result})
    except Exception as e:
        # Handle exceptions and return an error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

