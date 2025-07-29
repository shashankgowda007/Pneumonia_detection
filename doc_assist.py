import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
from openai import OpenAI
from cnn_baseline import PneumoniaDataset  # Reuse dataset class

# Removed OpenAI client initialization since we're using local knowledge only

# Constants
IMG_SIZE = (224, 224)  # ViT expects 224x224
CLASS_NAMES = ['Normal', 'Pneumonia']

# Load trained models
def load_model(model_path, model_type='vit'):
    if model_type == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        model.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2))
        # Load only matching keys
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)
    else:  # CNN
        from cnn_baseline import PneumoniaCNN
        model = PneumoniaCNN()
        # Load only matching keys
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Predict pneumonia
def predict_pneumonia(image, model_type='vit'):
    model = load_model(f'pneumonia_{model_type}.pth', model_type)
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(2)}

# Answer medical questions with comprehensive local knowledge
def answer_question(question, context=""):
    # Enhanced local knowledge base
    local_knowledge = {
        "types": {
            "keywords": ["type", "kind", "category", "classification"],
            "answer": "Main pneumonia types:\n"
                     "1. Bacterial (most common)\n"
                     "2. Viral (often from flu/RSV)\n"
                     "3. Fungal (in immunocompromised)\n"
                     "4. Aspiration (from inhaling substances)\n"
                     "5. Walking (mild bacterial cases)"
        },
        "treatment": {
            "keywords": ["treat", "cure", "medicine", "antibiotic", "heal", "therapy"],
            "answer": "Detailed treatment options:\n\n"
                     "🦠 Bacterial Pneumonia:\n"
                     "- First-line: Amoxicillin or Macrolides\n"
                     "- Severe cases: Fluoroquinolones + Cephalosporins\n"
                     "- Duration: 5-7 days typically\n\n"
                     "🦠 Viral Pneumonia:\n"
                     "- Antivirals for influenza (Tamiflu)\n"
                     "- Supportive care (oxygen, fluids)\n"
                     "- No antibiotics unless bacterial co-infection\n\n"
                     "🏥 Hospitalization needed when:\n"
                     "- Oxygen saturation <90%\n"
                     "- Confusion/disorientation\n"
                     "- Blood pressure <90/60\n"
                     "- Age >65 with comorbidities"
        },
        "symptoms": {
            "keywords": ["symptom", "sign", "feel", "experience"],
            "answer": "Common pneumonia symptoms:\n"
                     "- Cough (often with phlegm)\n"
                     "- Fever, chills, sweating\n"
                     "- Shortness of breath\n"  
                     "- Chest pain when breathing/coughing\n"
                     "- Fatigue, loss of appetite"
        },
        "duration": {
            "keywords": ["long", "last", "recover", "days"],
            "answer": "Recovery times:\n"
                     "- Mild cases: 1-2 weeks\n"
                     "- Severe cases: 3-6 weeks\n"
                     "- Elderly/compromised: May take months"
        },
        "contagious": {
            "keywords": ["spread", "contagious", "catch", "infect"],
            "answer": "Contagion facts:\n"
                     "- Bacterial/Viral: Contagious via droplets\n"
                     "- Most contagious during fever\n"
                     "- Contagious period: 1-2 days on antibiotics (bacterial)"
        },
        "prevention": {
            "keywords": ["prevent", "avoid", "vaccine", "stop"],
            "answer": "Prevention methods:\n"
                     "- Pneumococcal and flu vaccines\n"
                     "- Good hand hygiene\n"
                     "- Don't smoke\n"
                     "- Manage chronic conditions"
        },
        "clinical": {
            "keywords": ["clinical", "diagnosis", "diagnostic", "insight", "findings", "assessment"],
            "answer": "Clinical Insights:\n\n"
                     "🔍 Diagnostic Findings:\n"
                     "- Crackles/rales on auscultation\n"
                     "- Dullness to percussion\n"
                     "- Increased vocal fremitus\n"
                     "- Tachypnea (>20 breaths/min)\n\n"
                     "💊 Management Approach:\n"
                     "- Assess severity using CURB-65 score\n"
                     "- Obtain sputum culture if hospitalized\n"
                     "- Consider chest X-ray confirmation\n"
                     "- Monitor oxygen saturation\n\n"
                     "⚠️ Red Flags:\n"
                     "- Sepsis signs (fever, hypotension)\n"
                     "- Respiratory distress\n"
                     "- Altered mental status\n"
                     "- Failure to improve in 48-72h"
        }
    }

    # Find matching local knowledge
    question_lower = question.lower()
    for topic, data in local_knowledge.items():
        if any(keyword in question_lower for keyword in data["keywords"]):
            return f"Medical Knowledge:\n{data['answer']}"

    return ("I can answer questions about:\n"
            "- Pneumonia symptoms and signs\n"
            "- Treatment options and medications\n"  
            "- Recovery times and duration\n"
            "- Contagion and prevention\n\n"
            "Please ask specifically about these topics.")

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Pneumonia Detection Assistant")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Chest X-ray")
            model_type = gr.Radio(
                choices=["ViT", "CNN"], 
                value="ViT",
                label="Model Type"
            )
            predict_btn = gr.Button("Analyze")
        
        with gr.Column():
            label_output = gr.Label(label="Diagnosis Probability")
            plot_output = gr.Plot(label="Attention Map")
    
    with gr.Accordion("Ask a medical question"):
        question_input = gr.Textbox(label="Question")
        answer_output = gr.Textbox(label="Answer")
        ask_btn = gr.Button("Ask")
    
    # Event handlers
    predict_btn.click(
        fn=predict_pneumonia,
        inputs=[image_input, model_type],
        outputs=label_output
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(share=True)
