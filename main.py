import streamlit as st
import pickle

# Model aur Vectorizer load karein
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Fake News Detector")

# Input field
news_input = st.text_area("Paste the news content here:", key="unique_text_area")

# Button
if st.button("Verify News", key="verify_button"):
    if news_input.strip() != "":
        data = [news_input]
        vectorized_data = vectorizer.transform(data)
        prediction = model.predict(vectorized_data)
        
        # Result logic
        if prediction[0] == 'REAL':
            st.success("Real News! (Sachhi Khabar)")
        else:
            st.error("Fake News! (Jhoothi Khabar)")
    else:
        st.warning("Please paste some news first!")