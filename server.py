from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch
import torchvision


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# Face detection using OpenCV Haar Cascade (no additional installation needed)

# Load Haar Cascade for face detection (OpenCV built-in)
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    face_cascade = None
    print("Warning: Could not load Haar Cascade face detector")


from torch.autograd import Variable

import time

import sys


from torch import nn

from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Uploaded_Files')
video_path = ""

detectOutput = []

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creating Model Architecture

class Model(nn.Module):
  def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
    super(Model, self).__init__()

    # returns a model pretrained on ImageNet dataset
    model = models.resnext50_32x4d(pretrained= True)

    # Sequential allows us to compose modules nn together
    self.model = nn.Sequential(*list(model.children())[:-2])

    # RNN to an input sequence
    self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

    # Activation function
    self.relu = nn.LeakyReLU()

    # Dropping out units (hidden & visible) from NN, to avoid overfitting
    self.dp = nn.Dropout(0.4)

    # A module that creates single layer feed forward network with n inputs and m outputs
    self.linear1 = nn.Linear(2048, num_classes)

    # Applies 2D average adaptive pooling over an input signal composed of several input planes
    self.avgpool = nn.AdaptiveAvgPool2d(1)



  def forward(self, x):
    batch_size, seq_length, c, h, w = x.shape

    # new view of array with same data
    x = x.view(batch_size*seq_length, c, h, w)

    fmap = self.model(x)
    x = self.avgpool(fmap)
    x = x.view(batch_size, seq_length, 2048)
    x_lstm,_ = self.lstm(x, None)
    return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.squeeze()
  image = inv_normalize(image)
  image = image.numpy()
  image = image.transpose(1,2,0)
  image = image.clip(0,1)
  cv2.imwrite('./2.png', image*255)
  return image

# For prediction of output  
def predict(model, img, path='./'):
  # use this command for gpu    
  # fmap, logits = model(img.to('cuda'))
  fmap, logits = model(img.to())
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _, prediction = torch.max(logits, 1)
  confidence = logits[:, int(prediction.item())].item()*100
  print('confidence of prediction: ', logits[:, int(prediction.item())].item()*100)
  return [int(prediction.item()), confidence]


# To validate the dataset
class validation_dataset(Dataset):
  def __init__(self, video_names, sequence_length = 60, transform=None):
    self.video_names = video_names
    self.transform = transform
    self.count = sequence_length

  # To get number of videos
  def __len__(self):
    return len(self.video_names)

  # To get number of frames
  def __getitem__(self, idx):
    video_path = self.video_names[idx]
    frames = []
    a = int(100 / self.count)
    first_frame = np.random.randint(0,a)
    for i, frame in enumerate(self.frame_extract(video_path)):
      # Use OpenCV Haar Cascade for face detection (fallback to full frame if no face found)
      if face_cascade is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        try:
          x, y, w, h = faces[0]
          frame = frame[y:y+h, x:x+w, :]
        except:
          pass
      frames.append(self.transform(frame))
      if(len(frames) == self.count):
        break
    frames = torch.stack(frames)
    frames = frames[:self.count]
    return frames.unsqueeze(0)

  # To extract number of frames
  def frame_extract(self, path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
      success, image = vidObj.read()
      if success:
        yield image


def detectFakeImage(imagePath):
    try:
        im_size = 112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((im_size,im_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)])
        
        # Read the image
        image = cv2.imread(imagePath)
        if image is None:
            raise ValueError(f"Could not read image from {imagePath}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use face detection if available
        if face_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            try:
                x, y, w, h = faces[0]
                image = image[y:y+h, x:x+w, :]
            except:
                pass  # Use full image if no face detected
        
        # Transform the image
        transformed_image = train_transforms(image)
        
        # Create a sequence by duplicating the image (since model expects sequence)
        sequence_length = 20
        frames = [transformed_image] * sequence_length
        frames = torch.stack(frames)
        frames = frames.unsqueeze(0)  # Add batch dimension
        
        # Load model
        model = Model(2)
        path_to_model = os.path.join(BASE_DIR, 'model', 'df_model.pt')
        print(f"Loading model from: {path_to_model}")
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()
        
        # Make prediction
        prediction = predict(model, frames, './')
        
        return prediction
    except Exception as e:
        print(f"Error in detectFakeImage: {str(e)}")
        raise


@app.route('/', methods=['POST', 'GET'])
def homepage():
  if request.method == 'GET':
	  return render_template('index.html')
      # return render_template('index.html')


@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        try:
            image = request.files['image']
            print(f"Received image: {image.filename}")
            image_filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            print(f"Full target path: {os.path.abspath(image_path)}")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            image.save(image_path)
            print(f"Saved image to: {image_path}")
            
            # Release any file handles before prediction
            prediction = detectFakeImage(image_path)
            print(f"Prediction result: {prediction}")
            
            if prediction[0] == 0:
                  output = "FAKE"
            else:
                  output = "REAL"
            confidence = prediction[1]
            data = {'output': output, 'confidence': float(confidence)}
            
            # Try to remove the uploaded file with error handling
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Removed file: {image_path}")
            except PermissionError:
                print(f"Could not remove {image_path}, file may still be in use")
            except Exception as e:
                print(f"Error removing file: {e}")
                
            return jsonify(data)
        except Exception as e:
            print(f"Error in /Detect: {str(e)}")
            return jsonify({'error': str(e)}), 500
        

app.run(debug=True, port=5500)
