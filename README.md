# DEEP TB CARE AI
using RAG for LLM and LVMs for classification 
"DeepTBCareAI" has three main features, each with its own functionality and use case in the healthcare domain, especially focused on tuberculosis (TB) care.

1. Classify X-ray
Functionality:
This feature allows users to upload a chest X-ray image, which is then classified using a pre-trained deep learning model **EfficientNet**
to assess the risk of diseases like TB or related lung conditions.

If probabilities don't match the number of labels, an error is shown for safety.

2. 🤖 FAQ Chatbot
Functionality:
An AI chatbot built using **Cohere’s large language model** to answer TB-related FAQs, specifically focusing on nutrition, medication, and general TB care.



3. 🍎 Calorie Estimator
Functionality:
This module uses a food image classifier to identify the type of food in an uploaded image. It is designed to assist with nutritional monitoring, especially important for tuberculosis patients who require a well-balanced diet.

Current Capability:

Predicts the name of the food item from the image using a trained deep learning model  **Resnet 141** .

Future Scope:

Integration of volumetric segmentation techniques to estimate portion size and calorie content of the food, enabling more accurate dietary tracking.



To run:  streamlit Final_App.py    "Run this command in command prompt in the project environment"
