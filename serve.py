"""
Gradio Web Interface for Coffee Bean Classification
Supports both SimpleCNN and ResNet18 models
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse

from model import SimpleCNN, create_resnet18_finetuned


def load_models(cnn_path, resnet_path, device):
    """Load both trained models"""
    
    # Load SimpleCNN
    cnn_model = SimpleCNN().to(device)
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn_model.eval()
    print(f"✓ SimpleCNN loaded: {cnn_path}")
    
    # Load ResNet18
    resnet_model = create_resnet18_finetuned(num_classes=2, pretrained=False).to(device)
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model.eval()
    print(f"✓ ResNet18 loaded: {resnet_path}")
    
    return cnn_model, resnet_model


def create_interface(cnn_model, resnet_model, device, class_names=['Arabica', 'Robusta']):
    """Create Gradio interface"""
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def predict_cnn(image):
        """Predict using SimpleCNN"""
        if image is None:
            return {"Error": 1.0}
        
        img = Image.fromarray(image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
        
        return {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    
    def predict_resnet(image):
        """Predict using ResNet18"""
        if image is None:
            return {"Error": 1.0}
        
        img = Image.fromarray(image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = resnet_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
        
        return {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    
    def predict_both(image):
        """Predict using both models"""
        if image is None:
            return {"Error": 1.0}, {"Error": 1.0}
        return predict_cnn(image), predict_resnet(image)
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # ☕ Coffee Bean Classification
            ## Compare Custom CNN vs ResNet18 Fine-tuned
            Upload a coffee bean image to classify it as **Arabica** or **Robusta**
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Coffee Bean Image", type="numpy")
                
                with gr.Row():
                    btn_cnn = gr.Button("🔹 Predict with Custom CNN", variant="secondary")
                    btn_resnet = gr.Button("🔸 Predict with ResNet18", variant="primary")
                
                btn_both = gr.Button("⚡ Compare Both Models", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### 🔹 Custom 3-Layer CNN")
                output_cnn = gr.Label(label="CNN Prediction", num_top_classes=2)
                
                gr.Markdown("### 🔸 ResNet18 Fine-tuned")
                output_resnet = gr.Label(label="ResNet Prediction", num_top_classes=2)
        
        # Model Info
        with gr.Accordion("📊 Model Information", open=False):
            gr.Markdown(
                """
                ### Custom 3-Layer CNN
                - **Accuracy:** 56.41% (60% Arabica, 52.63% Robusta)
                - **Parameters:** 93,954
                - **Architecture:** 3 Conv layers + Global Average Pooling
                - **Training:** 10 epochs on 295 samples
                
                ### ResNet18 Fine-tuned
                - **Accuracy:** 71.79% (80% Arabica, 63.16% Robusta)
                - **Parameters:** 11.2M (8.4M trainable)
                - **Architecture:** ResNet18 with unfrozen layer4
                - **Training:** 20 epochs with dual learning rates
                - **Improvement:** +15.38% over custom CNN
                """
            )
        
        # Button actions
        btn_cnn.click(fn=predict_cnn, inputs=input_image, outputs=output_cnn)
        btn_resnet.click(fn=predict_resnet, inputs=input_image, outputs=output_resnet)
        btn_both.click(fn=predict_both, inputs=input_image, outputs=[output_cnn, output_resnet])
    
    return demo


def main():
    parser = argparse.ArgumentParser(description='Serve Coffee Bean Classifier')
    parser.add_argument('--cnn_model', type=str, default='coffee_model_3layer.pth',
                        help='Path to SimpleCNN model (default: coffee_model_3layer.pth)')
    parser.add_argument('--resnet_model', type=str, default='coffee_model_resnet18_finetuned.pth',
                        help='Path to ResNet18 model (default: coffee_model_resnet18_finetuned.pth)')
    parser.add_argument('--share', action='store_true',
                        help='Create public link (default: False)')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port number (default: 7860)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("🎨 CREATING GRADIO INTERFACE")
    print("="*70)
    
    # Load models
    cnn_model, resnet_model = load_models(args.cnn_model, args.resnet_model, device)
    
    # Create interface
    demo = create_interface(cnn_model, resnet_model, device)
    
    print("="*70)
    print("✅ Gradio interface created successfully!")
    print("="*70)
    
    # Launch
    demo.launch(share=args.share, server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
