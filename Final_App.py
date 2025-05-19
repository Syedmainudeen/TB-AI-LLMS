import streamlit as st
from contextlib import suppress

# Set page configuration
st.set_page_config(
    page_title="DeepTBCareAI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation if not set
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #E6B800 !important; /* Darker yellow background with medical symbols */
        background-image: url('https://www.transparenttextures.com/patterns/medical.png');
        background-repeat: repeat;
    }
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
        color: #1E3A8A !important;
    }
    .title-section {
        text-align: center;
        padding: 1rem 0;
        width: 100%;
    }
    .menu-container {
        justify-content: center;
        align-items: center;
        gap: 80px; /* Increased spacing for symmetry */
        margin: 3rem auto;
        width: 100%;
        max-width: 1200px;
    }
    .menu-item {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        width: 280px;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .menu-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .menu-description {
        font-size: 0.85rem;
        color: #4B5563;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
  import streamlit as st
  if st.session_state["page"] == "home":
      st.markdown('<div class="title-section">', unsafe_allow_html=True)
      st.markdown('<h1 class="main-header">🫁 DeepTBCareAI</h1>', unsafe_allow_html=True)
      st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #4B5563;">Advanced AI-powered tuberculosis care and management</p>', unsafe_allow_html=True)
      st.markdown('</div>', unsafe_allow_html=True)
        
      menu_items = [
          {"emoji": "🫁", "title": "Classify X-ray", "page": "x_ray"},
          {"emoji": "🤖", "title": "FAQ Chatbot", "page": "faq_chatbot"},
          {"emoji": "🍎", "title": "Calorie Estimator", "page": "calorie_estimator"}
      ]
        
      col1, col2, col3 = st.columns([3, 3, 1])
        
      with col1:
          if st.button(f"{menu_items[0]['emoji']} {menu_items[0]['title']}", key=menu_items[0]['title']):
              st.session_state["page"] = menu_items[0]['page']
        
      with col2:
          if st.button(f"{menu_items[1]['emoji']} {menu_items[1]['title']}", key=menu_items[1]['title']):
              st.session_state["page"] = menu_items[1]['page']
        
      with col3:
          if st.button(f"{menu_items[2]['emoji']} {menu_items[2]['title']}", key=menu_items[2]['title']):
              st.session_state["page"] = menu_items[2]['page']

      st.markdown("<hr>", unsafe_allow_html=True)
      st.markdown(
          '<p style="text-align: center; color: #6B7280; font-size: 0.8rem;">© 2025 DeepTBCareAI - AI-powered tuberculosis care and management</p>',
          unsafe_allow_html=True
      )
        
  elif st.session_state["page"] == "x_ray":  # Load x_ray.py instead of "Hello, World!"
      from keras.models import load_model
      from PIL import Image
      import numpy as np
      import pandas as pd
      import plotly.express as px

      from util import classify, set_background

      # Set title
      st.title('DeepTBCare AI: X-Ray Classifier Prototype')

      # Set header
      st.header('Upload Chest X-Ray image 👇')

      # Upload file
      file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

      # Load classifier
      model = load_model('CovidModelEfficientNet.h5')

      # Load class names
      with open('labels.txt', 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]

      # Display image
      if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # Classify image
        class_probs = classify(image, model, class_names)
      
        # Ensure correct shape
        if  len(class_probs)!= len(class_names):
            st.error(f"Error: Model returned {len(class_probs)} probabilities, but there are {len(class_names)} classes.")
        else:
            # Create DataFrame
            df = pd.DataFrame({'Class': class_names, 'Probability': class_probs})

            # Horizontal bar chart
            fig = px.bar(df, y='Class', x='Probability', orientation='h',
                        title='Risk of disease', text_auto='.2%',
                        labels={'Percentage': 'disease'})
            st.plotly_chart(fig)
        
        if st.button("Back to Home"):
          st.session_state["page"] = "home"

    
  elif st.session_state["page"] == "calorie_estimator":  # Load x_ray.py instead of "Hello, World!"
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    from keras.preprocessing import image
    from PIL import Image
    # Load your pre-trained model
    model_path = 'final_food_classifier.h5'
    model = tf.keras.models.load_model(model_path)

    # Reading Class Names
    df=pd.read_csv('food_class_names.csv')
    label=df['Label']
    # Streamlit app
    st.title("🥙🥙 Food Classification Prototype")
    uploaded_file = None

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    # Make prediction on the uploaded image
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        img = Image.open(uploaded_file).resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        predictions=model.predict(img_array)
        # Display the top prediction
        st.subheader("Prediction:")
        st.write(label[np.argmax(predictions)])
      
    if st.button("Back to Home"):
      st.session_state["page"] = "home"
    
  elif st.session_state["page"] == "faq_chatbot":
    # pip install cohere
    import cohere
    import streamlit as st

    # Get your free API key: https://dashboard.cohere.com/api-keys
    co = cohere.ClientV2(api_key="nz4rF83AuH1s5wBdDkhGbyC3mr6JSX7po0V4YkZf")

    # Add the user message
    message = "I am building a Tuberculosis FAQ chatbot which will answer FAQs regarding nutrition and medication concisely."

    # text generator function
    def generate_response(user_input):
        if not user_input.strip():  # Handle empty input
            return "Please enter a prompt to get a response."

        response = co.chat(
          model="command-r-plus-08-2024",
          messages=[{"role": "user", "content": user_input},
          {"role": "system", "content": "Answer concisely, in **5 complete bullet points** and always ensure that there is a full stop at the end."},
          {"role": "system", "content": "If and Only if the user says 'Hello!' reply with a simple 'Hi!' and waut for furthur questions."} ],
          max_tokens=100,
        )
        st.write(response.message.content[0].text)

    # Streamlit UI
    st.title("🫁Tuberculosis FAQ Chatbot🤖")
    st.header("Ask me questions about tuberculosis, including nutrition and medication 👇")

    user_input = st.text_input("Enter your question:")

    if user_input.strip():
        if st.button("Generate Response"):
            response = generate_response(user_input)
            st.write(response)
    else:
        st.warning("Please enter a prompt before submitting.")
      
    if st.button("Back to Home"):
        st.session_state["page"] = "home"

if __name__ == "__main__":
    main()
