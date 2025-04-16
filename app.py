import os
from flask import Flask, render_template, request, redirect, url_for, session
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for login sessions

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ViT model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=4,
    ignore_mismatched_sizes=True
)
model = model.to(device)

# Load trained model weights
checkpoint = torch.load('lung_cancer_vit_model.pth', map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Clean class labels
class_names = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'normal',
    'squamous.cell.carcinoma'
]

# Dos and Don'ts for each class
dos_and_donts = {
    'adenocarcinoma': {
        'dos': [
            'Follow up regularly with your oncologist.',
            'Maintain a healthy diet rich in fruits and vegetables.',
            'Take medications and treatments as prescribed.'
        ],
        'donts': [
            'Avoid smoking or being around smoke.',
            'Don’t skip follow-up appointments.',
            'Avoid self-medicating or delaying treatment.'
        ]
    },
    'large.cell.carcinoma': {
        'dos': [
            'Seek prompt and aggressive treatment options.',
            'Consider getting a second opinion on treatment plans.',
            'Report any new symptoms to your doctor immediately.'
        ],
        'donts': [
            'Avoid delaying therapy or treatment sessions.',
            'Don’t ignore signs like persistent coughing or fatigue.',
            'Refrain from unhealthy lifestyle habits like smoking.'
        ]
    },
    'normal': {
        'dos': [
            'Continue a healthy lifestyle.',
            'Have regular health checkups.',
            'Stay active and eat balanced meals.'
        ],
        'donts': [
            'Don’t start smoking or using tobacco.',
            'Avoid air pollution exposure when possible.',
            'Don’t ignore early warning signs of respiratory issues.'
        ]
    },
    'squamous.cell.carcinoma': {
        'dos': [
            'Work with a specialist for a smoking cessation plan.',
            'Stick to your prescribed treatment plan.',
            'Track your symptoms and report changes.'
        ],
        'donts': [
            'Don’t continue smoking or using tobacco.',
            'Avoid skipping radiation or chemotherapy sessions.',
            'Don’t ignore persistent coughs or chest pain.'
        ]
    }
}

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor).logits
        _, predicted_class = torch.max(outputs, dim=1)
    return predicted_class.item()

# Routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['user'] = username
            return redirect(url_for('predict'))
        else:
            error = 'Invalid Credentials'
    return render_template('login.html', error=error)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['image']
        if file:
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            image_path = os.path.join(upload_folder, file.filename)
            file.save(image_path)

            predicted_class_index = predict_image(image_path)
            predicted_label = class_names[predicted_class_index]

            dos = dos_and_donts[predicted_label]['dos']
            donts = dos_and_donts[predicted_label]['donts']

            return render_template(
                'result.html',
                predicted_class=predicted_label,
                image_path=image_path,
                dos=dos,
                donts=donts
            )

    return render_template('predict.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
