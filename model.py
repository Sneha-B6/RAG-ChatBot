import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline

# Step 1: Load data from SQLite into pandas DataFrame
conn = sqlite3.connect('my_database.db')
df = pd.read_sql_query("SELECT * FROM imdb_reviews", conn)

# Step 2: Preprocessing - Split data into features and labels
X = df['review_text']
y = df['sentiment']

# Split the data into training and testing sets (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Vectorization and Model Training
# Use TF-IDF vectorizer with Logistic Regression
vectorizer = TfidfVectorizer(stop_words='english')
model = LogisticRegression()

pipeline = make_pipeline(vectorizer, model)

# Train the model
pipeline.fit(X_train, y_train)

# Step 4: Evaluate the model on the validation set
y_val_pred = pipeline.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_val_pred)
precision = precision_score(y_test, y_val_pred, pos_label='Positive')
recall = recall_score(y_test, y_val_pred, pos_label='Positive')
f1 = f1_score(y_test, y_val_pred, pos_label='Positive')

# Print validation set performance
print(f"Validation Set Performance:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}")

# Step 5: Store predictions in the database
# Check if the 'predicted_sentiment' column exists in the table
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(imdb_reviews)")
columns = [column[1] for column in cursor.fetchall()]

# Add the 'predicted_sentiment' column if it doesn't already exist
if 'predicted_sentiment' not in columns:
    cursor.execute('''
        ALTER TABLE imdb_reviews ADD COLUMN predicted_sentiment TEXT;
    ''')
    conn.commit()

# Store the predicted sentiments in the database for the test set
for idx, review in X_test.items():  # Use .items() instead of .iteritems()
    if idx < len(y_val_pred):  # Ensure idx is within bounds for y_val_pred
        predicted_label = y_val_pred[idx]
        cursor.execute('''
            UPDATE imdb_reviews
            SET predicted_sentiment = ?
            WHERE review_text = ?
        ''', (predicted_label, review))

conn.commit()



# Step 6: Close the connection to the database
conn.close()

# Testing: After running the above code, check if the 'predicted_sentiment' column has been updated
