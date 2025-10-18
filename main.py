import re
import json
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from openai import OpenAI
from dotenv import load_dotenv
import os

# -------------------- 🔐 Load API Key from .env --------------------
load_dotenv()  # Load environment variables from .env
API_KEY = os.getenv("OPENAI_API_KEY")

# Use client only if API key exists
client = None
if not API_KEY:
    st.warning("⚠️ OpenAI API Key not found. Please set OPENAI_API_KEY in .env file.")
else:
    client = OpenAI(api_key=API_KEY)

# -------------------- 📚 NLTK Setup --------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# -------------------- ⚙️ Streamlit Config --------------------
st.set_page_config(page_title="Mental Health Detector (GPT)", page_icon="🧠", layout="centered")

# -------------------- 🧼 Text Cleaning --------------------
def clean_statement(statement: str) -> str:
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = [w for w in statement.split() if w not in stop_words]
    return ' '.join(words)

# -------------------- ✨ Highlight Keywords --------------------
def highlight_keywords(text, keywords):
    for word in keywords:
        text = re.sub(f"(?i)\\b{word}\\b", f"**{word}**", text)
    return text

# -------------------- 🤖 OpenAI Classification --------------------
def detect_mental_state_openai(text: str, clean_text: bool = True):
    if client is None:
        # Return dummy values if API key missing
        return "API Key Missing", 0, ["API Key Missing"]

    if clean_text:
        text = clean_statement(text)

    categories = [
        "Depression",
        "Anxiety",
        "Stress",
        "Suicidal Thoughts",
        "Happy / Positive",
        "Neutral"
    ]

    system_prompt = f"""
    You are a mental health text classifier.
    Classify the following text into one of these categories:
    {', '.join(categories)}.

    Return your response in JSON format with the fields:
    - "label": the predicted category
    - "confidence": a number between 0 and 100 representing how confident you are.
    """

    user_prompt = f"Text: {text}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    try:
        data = json.loads(content)
        label = data.get("label", "Unknown")
        confidence = float(data.get("confidence", 0))
    except Exception:
        label = content
        confidence = 0

    return label, confidence, categories

# -------------------- 📌 Sidebar Menu --------------------
menu = ["Home", "About", "FAQ"]
choice = st.sidebar.selectbox("Menu", menu)

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- 🏠 HOME PAGE --------------------
if choice == "Home":
    st.title("🧠 Mental Health Status Detection")
    st.write("This app uses **OpenAI GPT** to classify your mental health status from text.")

    input_text = st.text_area("📝 Enter your description:", height=120)
    clean_option = st.checkbox("Clean input text before analysis", value=True)

    if st.button("Analyze"):
        if not input_text.strip():
            st.error("Please enter some text before detection.")
        else:
            with st.spinner("🔍 Analyzing your text using GPT..."):
                try:
                    label, confidence, categories = detect_mental_state_openai(
                        input_text, clean_text=clean_option
                    )

                    st.success(f"Predicted Mental Status: **{label}**")
                    st.progress(int(confidence))

                    # Show confidence distribution (mock for visualization)
                    df = pd.DataFrame({
                        "Class": categories,
                        "Confidence": [confidence if c == label else (100 - confidence) / (len(categories)-1) for c in categories]
                    })
                    st.bar_chart(df.set_index("Class"))

                    # Keyword Highlight
                    top_words = [w for w in input_text.split() if w.lower() in clean_statement(input_text).split()]
                    highlighted_text = highlight_keywords(input_text, top_words)
                    st.markdown(f"**Highlighted keywords:** {highlighted_text}")

                    # Save history
                    st.session_state.history.append({
                        "Text": input_text,
                        "Prediction": label,
                        "Confidence": confidence
                    })

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Show History
    if st.session_state.history:
        st.subheader("📜 Previous Predictions")
        for item in reversed(st.session_state.history):
            st.write(f"**Text:** {item['Text']}")
            st.write(f"**Prediction:** {item['Prediction']} (Confidence: {item['Confidence']:.1f}%)")
            st.markdown("---")

# -------------------- ℹ️ ABOUT PAGE --------------------
elif choice == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This app uses **OpenAI GPT-3.5** to detect mental health signals in text.  
    🧠 No local model needed  
    ⚡ Real-time responses from OpenAI  
    ❌ Not a medical tool — for educational use only
    """)

# -------------------- ❓ FAQ PAGE --------------------
elif choice == "FAQ":
    st.title("❓ FAQ")
    st.markdown("""
    **Q: What type of input works best?**  
    A: Full sentences that describe feelings. E.g., “I feel hopeless and tired lately.”

    **Q: What does confidence mean?**  
    A: The model’s self-estimated certainty about the classification.

    **Q: Is this a medical tool?**  
    A: ❌ No — for educational & research purposes only.
    """)

# -------------------- Footer --------------------
st.markdown("---")
st.caption("© 2025 Mental Health Detector (OpenAI GPT-3.5) | Educational Use Only")
