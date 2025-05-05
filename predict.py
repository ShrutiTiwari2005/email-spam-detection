import pickle

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# User input
message = input("Enter email text: ")
message_vector = vectorizer.transform([message])

# Predict
result = model.predict(message_vector)
print("Result:", "SPAM" if result[0] else "NOT SPAM")
