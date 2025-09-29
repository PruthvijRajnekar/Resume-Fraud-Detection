from flask import Flask, request, jsonify
from resume_fraud_detector import predict_resume_fraud  # your existing detector function
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from frontend (for local testing)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')

        if not resume_text.strip():
            return jsonify({"error": "No resume text provided"}), 400

        # Call your existing fraud detection function
        # Make sure predict_resume_fraud returns a dictionary like this:
        # {
        #     "risk": 75,
        #     "risk_label": "High",
        #     "word_count": 450,
        #     "experience_years": 2,
        #     "readability_score": 70,
        #     "inconsistencies": 3,
        #     "red_flags": ["Fake degree", "Inconsistent dates"],
        #     "recommendations": ["Verify education", "Check work experience"]
        # }
        result = predict_resume_fraud(resume_text)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
