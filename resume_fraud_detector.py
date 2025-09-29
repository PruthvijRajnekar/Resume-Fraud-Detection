import pandas as pd
import numpy as np
import re
from datetime import datetime
from dateutil import parser
import nltk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from textstat import flesch_reading_ease, flesch_kincaid_grade
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class ResumeProcessor:
    """Class to process and extract features from resumes"""
    
    def __init__(self):
        self.education_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'diploma']
        self.skill_keywords = ['python', 'java', 'javascript', 'sql', 'machine learning', 'data science']
        self.suspicious_patterns = [
            'world-class', 'best in class', 'expert level', 'guru', 'ninja', 'rockstar',
            'thought leader', 'industry expert', 'revolutionary', 'groundbreaking'
        ]
    
    def extract_dates(self, text):
        """Extract dates from resume text"""
        date_patterns = [
            r'\b\d{4}\b',  # Year only
            r'\d{1,2}/\d{4}',  # Month/Year
            r'\d{1,2}-\d{4}',  # Month-Year
            r'[A-Za-z]+ \d{4}'  # Month Year
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return dates
    
    def calculate_experience_years(self, text):
        """Calculate total years of experience"""
        dates = self.extract_dates(text)
        years = []
        
        for date in dates:
            try:
                if re.match(r'\b\d{4}\b', date):
                    years.append(int(date))
                elif '/' in date or '-' in date:
                    year = int(date.split('/')[-1] if '/' in date else date.split('-')[-1])
                    years.append(year)
            except:
                continue
        
        if len(years) >= 2:
            return max(years) - min(years)
        return 0
    
    def extract_text_features(self, text):
        """Extract various text-based features"""
        features = {}
        
        # Basic text statistics
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Readability scores
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
        except:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
        
        # Keyword counts
        text_lower = text.lower()
        features['education_keywords'] = sum([text_lower.count(keyword) for keyword in self.education_keywords])
        features['skill_keywords'] = sum([text_lower.count(keyword) for keyword in self.skill_keywords])
        features['suspicious_patterns'] = sum([text_lower.count(pattern) for pattern in self.suspicious_patterns])
        
        # Experience calculation
        features['experience_years'] = self.calculate_experience_years(text)
        
        # Email and phone patterns
        features['email_count'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        features['phone_count'] = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
        
        # Repetitive content detection
        words = text.lower().split()
        word_freq = Counter(words)
        features['most_common_word_freq'] = word_freq.most_common(1)[0][1] if word_freq else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        
        return features
    
    def detect_inconsistencies(self, text):
        """Detect potential inconsistencies in resume"""
        inconsistencies = 0
        
        # Check for date inconsistencies
        dates = self.extract_dates(text)
        years = []
        for date in dates:
            try:
                if re.match(r'\b\d{4}\b', date):
                    years.append(int(date))
            except:
                continue
        
        # Flag if there are future dates
        current_year = datetime.now().year
        future_dates = [year for year in years if year > current_year]
        inconsistencies += len(future_dates)
        
        # Check for unrealistic experience claims
        if self.extract_text_features(text)['experience_years'] > 50:
            inconsistencies += 1
        
        return inconsistencies

class FraudDetector:
    """Main fraud detection class"""
    
    def __init__(self):
        self.processor = ResumeProcessor()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42)
        }
        self.trained_models = {}
        
    def preprocess_data(self, resumes, labels=None):
        """Preprocess resume data and extract features"""
        print("Preprocessing resumes...")
        
        # Extract text features for each resume
        text_features_list = []
        processed_texts = []
        
        for resume in resumes:
            # Clean text
            cleaned_text = re.sub(r'[^\w\s]', ' ', str(resume))
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            processed_texts.append(cleaned_text)
            
            # Extract features
            features = self.processor.extract_text_features(cleaned_text)
            features['inconsistencies'] = self.processor.detect_inconsistencies(cleaned_text)
            text_features_list.append(features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(text_features_list)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Create TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                               columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Combine all features
        final_features = pd.concat([feature_df, tfidf_df], axis=1)
        
        return final_features, processed_texts
    
    def train_models(self, X, y):
        """Train multiple models for fraud detection"""
        print("Training models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train supervised models
        for name, model in self.models.items():
            if name != 'isolation_forest':  # Skip unsupervised model
                print(f"Training {name}...")
                model.fit(X_scaled, y)
                self.trained_models[name] = model
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
                print(f"{name} - CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train anomaly detection model (unsupervised)
        isolation_forest = self.models['isolation_forest']
        isolation_forest.fit(X_scaled)
        self.trained_models['isolation_forest'] = isolation_forest
        
        print("Model training completed!")
    
    def predict_fraud(self, X):
        """Make predictions using trained models"""
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, model in self.trained_models.items():
            if name == 'isolation_forest':
                # Isolation Forest returns -1 for outliers, 1 for inliers
                pred = model.predict(X_scaled)
                predictions[name] = (pred == -1).astype(int)  # Convert to 1 for fraud, 0 for legitimate
            else:
                predictions[name] = model.predict_proba(X_scaled)[:, 1]  # Probability of fraud
        
        return predictions
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nModel Evaluation:")
        print("=" * 50)
        
        results = {}
        
        for name, model in self.trained_models.items():
            if name != 'isolation_forest':
                y_pred = model.predict(self.scaler.transform(X_test))
                y_prob = model.predict_proba(self.scaler.transform(X_test))[:, 1]
                
                auc_score = roc_auc_score(y_test, y_prob)
                
                print(f"\n{name.upper()} Results:")
                print(f"AUC Score: {auc_score:.3f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                results[name] = {
                    'auc': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_prob
                }
        
        return results
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance for Random Forest"""
        if 'random_forest' in self.trained_models:
            rf_model = self.trained_models['random_forest']
            feature_importance = rf_model.feature_importances_
            
            # Get top 20 features
            top_indices = np.argsort(feature_importance)[-20:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Most Important Features for Fraud Detection')
            plt.tight_layout()
            plt.show()

# Example usage and demo
def create_sample_data():
    """Create sample resume data for demonstration"""
    
    # Sample legitimate resumes
    legitimate_resumes = [
        """John Smith
        Email: john.smith@email.com
        Phone: 555-123-4567
        
        Experience:
        Software Engineer at Tech Corp (2020-2023)
        - Developed web applications using Python and JavaScript
        - Collaborated with team of 5 developers
        
        Junior Developer at StartUp Inc (2018-2020)
        - Built REST APIs using Django framework
        - Participated in agile development process
        
        Education:
        Bachelor of Science in Computer Science
        State University (2014-2018)
        
        Skills: Python, JavaScript, SQL, Git""",
        
        """Mary Johnson
        Email: mary.johnson@email.com
        Phone: 555-987-6543
        
        Experience:
        Data Analyst at Analytics Co (2021-2023)
        - Analyzed customer data using Python and SQL
        - Created dashboards using Tableau
        
        Research Assistant at University Lab (2019-2021)
        - Conducted statistical analysis of research data
        - Published 2 papers in peer-reviewed journals
        
        Education:
        Master of Science in Statistics
        Research University (2017-2019)
        
        Bachelor of Mathematics
        College State (2013-2017)
        
        Skills: Python, R, SQL, Tableau, Statistics"""
    ]
    
    # Sample fraudulent resumes (with red flags)
    fraudulent_resumes = [
        """Alex World-Class Expert
        Email: alex.guru@email.com
        Phone: 555-000-0000
        
        Experience:
        Senior Architect and Industry Expert at Global Tech (2025-2030)
        - Revolutionary groundbreaking work in AI and machine learning
        - Led team of 100+ world-class engineers
        - Generated $10 billion in revenue single-handedly
        
        CTO and Thought Leader at Innovation Corp (2020-2025)
        - Built industry-leading products used by millions
        - Expert level in all programming languages
        - Recognized as top 1% developer globally
        
        Education:
        PhD in Computer Science from Harvard (2018-2020)
        Master of Science from MIT (2016-2018)
        Bachelor from Stanford (2012-2016)
        
        Skills: Expert in Python, Java, C++, JavaScript, Go, Rust, AI, ML, Blockchain, 
        Quantum Computing, and 50+ other technologies""",
        
        """Bob Suspicious Resume
        Email: bob.fake@email.com
        
        Experience:
        CEO and Founder of Multiple Fortune 500 Companies (2022-2023)
        - Best in class performance across all metrics
        - Ninja-level problem solving abilities
        - Rockstar developer and business leader
        
        Senior Everything at Tech Giant (2021-2022)
        - Did everything and anything required
        - Expert level proficiency in all technologies
        - Generated unlimited revenue and growth
        
        Education:
        PhD from Top University (2020-2021)
        Master from Another Top School (2019-2020)
        
        Skills: Everything, All programming languages, Business, Leadership,
        Machine Learning Guru, Data Science Ninja, Full-stack Rockstar"""
    ]
    
    # Combine data
    all_resumes = legitimate_resumes + fraudulent_resumes
    labels = [0] * len(legitimate_resumes) + [1] * len(fraudulent_resumes)
    
    return all_resumes, labels

def main():
    """Main function to demonstrate the fraud detection system"""
    print("Resume Fraud Detection System")
    print("=" * 40)
    
    # Create sample data
    resumes, labels = create_sample_data()
    print(f"Loaded {len(resumes)} sample resumes ({sum(labels)} fraudulent, {len(labels)-sum(labels)} legitimate)")
    
    # Initialize fraud detector
    detector = FraudDetector()
    
    # Preprocess data
    features, processed_texts = detector.preprocess_data(resumes, labels)
    print(f"Extracted {features.shape[1]} features from resumes")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.5, random_state=42, stratify=labels
    )
    
    # Train models
    detector.train_models(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    predictions = detector.predict_fraud(X_test)
    
    # Display predictions
    print("\nPrediction Results:")
    print("-" * 30)
    for i, (actual, resume) in enumerate(zip(y_test, [resumes[j] for j in X_test.index])):
        print(f"\nResume {i+1} (Actual: {'Fraudulent' if actual == 1 else 'Legitimate'}):")
        print(f"First 100 characters: {resume[:100]}...")
        
        for model_name, pred in predictions.items():
            if model_name == 'isolation_forest':
                print(f"{model_name}: {'Fraudulent' if pred[i] == 1 else 'Legitimate'}")
            else:
                fraud_prob = pred[i]
                print(f"{model_name}: {fraud_prob:.3f} (fraud probability)")
        print("-" * 30)
    
    # Evaluate models (if we have enough data)
    if len(set(y_test)) > 1:  # Check if we have both classes in test set
        results = detector.evaluate_models(X_test, y_test)
    
    # Plot feature importance
    detector.plot_feature_importance(features.columns.tolist())
    
    print("\nFraud detection analysis completed!")
    
    return detector, features, labels

if __name__ == "__main__":
    detector, features, labels = main()

# Function to use in Flask API
def predict_resume_fraud(resume_text):
    processor = ResumeProcessor()
    features = processor.extract_text_features(resume_text)
    features['inconsistencies'] = processor.detect_inconsistencies(resume_text)
    
    # Simplified scoring for demo
    risk_score = (
        features['suspicious_patterns'] * 10 +
        features['inconsistencies'] * 20 +
        (features['experience_years'] < 1) * 10
    )
    risk_score = min(risk_score, 100)  # cap at 100
    
    # Determine risk label
    if risk_score >= 70:
        risk_label = "High"
    elif risk_score >= 40:
        risk_label = "Medium"
    else:
        risk_label = "Low"
    
    # Return dictionary compatible with frontend
    return {
        "risk": risk_score,
        "risk_label": risk_label,
        "word_count": features['word_count'],
        "experience_years": features['experience_years'],
        "readability_score": features['flesch_reading_ease'],
        "inconsistencies": features['inconsistencies'],
        "red_flags": ["Suspicious keywords detected"] if features['suspicious_patterns'] > 0 else [],
        "recommendations": ["Verify experience and education"] if features['inconsistencies'] > 0 else []
    }
