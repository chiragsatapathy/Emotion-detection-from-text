
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import streamlit as st
import altair as alt

# Load and preprocess the dataset
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("emotion_dataset.csv")
    df["Clean_Text"] = df["Text"].apply(nfx.remove_stopwords)
    x = df["Clean_Text"]
    y = df["Emotion"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    pipe_lr = Pipeline(steps=[("cv", CountVectorizer()), ("lr", LogisticRegression())])
    pipe_lr.fit(x_train, y_train)
    joblib.dump(pipe_lr, "text_emotion.pkl")
    return pipe_lr

# Load model if already trained, else train
try:
    pipe_lr = joblib.load("text_emotion.pkl")
except:
    pipe_lr = load_and_train_model()

# Emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# Prediction functions
def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# Streamlit app
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key="my_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        col1, col2 = st.columns(2)
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.write(f"{prediction}:{emoji_icon}")
            st.write(f"Confidence: {np.max(probability)}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]
            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x="emotions", y="probability", color="emotions"
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
