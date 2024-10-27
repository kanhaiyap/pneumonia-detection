Pneumonia Detection Using Computer Vision
Overview
This project aims to develop a state-of-the-art computer vision model for detecting pneumonia from chest X-ray images. By leveraging advanced techniques in deep learning, particularly Convolutional Neural Networks (CNNs), the model can accurately classify X-ray images into Pneumonia and Normal categories.
Problem Statement
Pneumonia is a serious lung infection that can lead to severe health complications if not diagnosed promptly. Traditional diagnostic methods can be slow, making it crucial to develop a fast and reliable automated detection system using X-ray images.
Dataset
The dataset consists of 5,863 chest X-ray images categorized into two classes:
Pneumonia
Normal

Model Architecture
This project employs a CNN architecture, which has proven effective for image classification tasks. The following models were tested:
ResNet50
DenseNet121
InceptionV3
Performance Metrics
The models were evaluated based on accuracy and loss metrics. Below is a summary of the performance:
Model	Train Accuracy	Test Accuracy	Loss
ResNet50	91.26%	76.76%	0.3043
DenseNet121	88.92%	73.40%	0.3211
InceptionV3	86.22%	87.18%	0.2765
Installation
To set up the environment, clone this repository and install the required packages:
bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
pip install -r requirements.txt

Usage
To run the model on your own dataset, follow these steps:
Place your X-ray images in the appropriate directories.
Execute the training script:
bash
python train.py --data_dir data/train --epochs 50 --batch_size 32

For predictions on new images, use:
bash
python predict.py --image_path path/to/your/xray.jpg

Future Work
Future enhancements may include:
Implementing an ensemble model for improved accuracy.
Exploring transfer learning with more advanced architectures.
Developing a web application for user-friendly access to the detection system.
Contributors
Kunwar Kanhaiya Kamlakant Pandey
