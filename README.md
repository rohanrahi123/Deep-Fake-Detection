# DeepFake Detection - Simplified!
DeepFake Detection Web-App 🖥 using Deep Learning(ResNext and LSTM), Flask and ReactJs where you can predict whether a video is FAKE Or REAL along with the confidence ratio. 

## Explanation of the Project
- We've created a DeepFake Detection system which intends to detect DeepFake videos using Deep Learning techniques like ResNext and LSTM. Also integrated the  trained model with the Frontend UI which uses ReactJs and Backend uses Flask.

- The Dataset we've used to train our model is available online.

- To find our trained model, check the model folder.

- To understand the project in a better way it is structured in below format:
```
DeepFake-Detection
    |
    |--- DeepFake_Detection
    |--- Implementation Video
    |--- Project-Setup.txt
    |--- Requiremnts.txt
```
1. DeepFake_Detection - This is the root folder.

2. Implementation Video - It shows the whole working of the project. 

3. Project-Setup.txt - In this file we've written all the necessary steps to run this project.

4. Requirements.txt - Python libraries needed for this project. 

## Project Set-up Guidelines
To set up the project. All the steps and guidelines regarding that are listed in Project-Setup.txt.

## Note
1. In the root folder(DeepFake_Detection), create a new folder called "Uploaded_Files".

2. In the root folder(DeepFake_Detection), create a new folder called "model" and add the model file in it.

<b>Add these folders to the root folder(DeepFake_Detection). Since, the path has already been given to the "server.py" file and also to avoid any path related errors.</b>

## Our Results

1) Accuracy of the Model:
<img width="250" height="50" alt="Model Accuracy" src="https://user-images.githubusercontent.com/58872872/133935912-1def7615-6538-4c88-9134-8f94a9367965.png">

2) Training and Validation Accuracy Graph:
<img width="378" alt="Accuracy Graph" src="https://user-images.githubusercontent.com/58872872/133936040-4bfa44a7-45c5-499b-8a10-f253cbcab56c.png">

3) Training and Validation Loss Graph:
<img width="381" alt="Loss Graph" src="https://user-images.githubusercontent.com/58872872/133935983-b4d9275f-e841-4b69-86cd-79c770ea2aa1.png">

4) Confusion Matrix:
<img width="402" alt="Confusion Matrix" src="https://user-images.githubusercontent.com/58872872/133936080-d2b39804-4a99-47b8-8be4-87ba77161961.png">

- To see the working of the project, check the Implementation Video.mp4.
