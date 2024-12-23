from flask import Flask, request, jsonify
import pickle

# Load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'Invalid input'}), 400

        message = data['message']
        vectorized_message = vectorizer.transform([message])
        prediction = model.predict(vectorized_message)[0]

        result = 'Spam' if prediction == 1 else 'Not Spam'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

