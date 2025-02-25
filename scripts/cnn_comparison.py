import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import numpy as np
import glob
import os
from collections import defaultdict
from ptflops import get_model_complexity_info
import time
from datetime import datetime
import csv

class InferencePipeline:
    def __init__(self, model_name='resnet50', device='cpu'):
        self.device = device
        self.model_name = model_name
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.class_labels = self._load_class_mapping()
        
    def _load_class_mapping(self):
        with open('imagenet1000_clsidx_to_labels.txt', 'r') as f:
            content = eval(f.read())
            return {int(k): v.split(',')[0].strip() for k, v in content.items()}
            
    def _load_model(self):
        # Dictionary of available models with their loading functions
        model_configs = {
            # Smaller models
            'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
            'shufflenet_v2_x0_5': (models.shufflenet_v2_x0_5, models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
            'squeezenet1_0': (models.squeezenet1_0, models.SqueezeNet1_0_Weights.IMAGENET1K_V1),
            
            # Medium-sized models
            'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
            'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
            'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
            'densenet121': (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
            
            # Larger models
            'vgg16': (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
            'vgg19': (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1),
            'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
            'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
            'densenet201': (models.densenet201, models.DenseNet201_Weights.IMAGENET1K_V1),
            'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            'efficientnet_b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.IMAGENET1K_V1),
        }
        
        if self.model_name not in model_configs:
            raise ValueError(f"Unsupported model: {self.model_name}. Available models: {list(model_configs.keys())}")
        
        model_fn, weights = model_configs[self.model_name]
        model = model_fn(weights=weights)
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, top_k=5):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=top_k)
            
            labels = [self.class_labels[idx.item()] for idx in top_indices[0]]
            probs = top_probs[0].cpu().numpy() * 100

            return {
                'probabilities': probs,
                'labels': labels,
                'class_name': self._extract_class_name(image_path)
            }

    def _extract_class_name(self, image_path):
        filename = os.path.basename(image_path)
        return filename.split('_')[1].split('.')[0]

    def batch_predict(self, num_images=50):
        image_paths = glob.glob(os.path.join('imagenet-samples', 'n*_*.JPEG'))
        # Take a random sample of images if there are more than num_images
        if len(image_paths) > num_images:
            image_paths = np.random.choice(image_paths, num_images, replace=False)
        
        results = defaultdict(list)
        inference_times = []
        predictions = []
        
        for path in image_paths:
            start_time = time.time()
            prediction = self.predict(path)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            true_class = self._extract_class_name(path)
            pred_class = prediction['labels'][0]
            results['correct'].append(pred_class.lower() == true_class.lower())
            
            predictions.append({
                'file': os.path.basename(path),
                'true_class': true_class,
                'top_pred': pred_class,
                'confidence': prediction['probabilities'][0]
            })
        
        return {
            'total_images': len(image_paths),
            'accuracy': sum(results['correct']) / len(results['correct']) * 100,
            'predictions': predictions,
            'avg_inference_time': np.mean(inference_times),
            'total_time': sum(inference_times)
        }

def main():
    total_start_time = time.time()
    timing_data = []

    models_to_test = [
        'shufflenet_v2_x0_5',
        'mobilenet_v2',
        'efficientnet_b0',
        'squeezenet1_0',
        'resnet18',
        'densenet121',
        'resnet152',
        'vgg16',
        'vgg19'
    ]
    
    print("Model Comparison:")
    print("-" * 100)
    print(f"{'Model':<20} {'Parameters':<15} {'FLOPs':<15} {'Avg Time/img(s)':<15} {'Total Time(s)':<15}")
    print("-" * 100)
    
    for model_name in models_to_test:
        time.sleep(10)  # Short pause between models
        
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        model_start_time = time.time()
        
        pipeline = InferencePipeline(model_name=model_name, device='cpu')
        
        # Get model complexity info
        macs, params = get_model_complexity_info(
            pipeline.model, 
            (3, 224, 224), 
            as_strings=False,
            print_per_layer_stat=False
        )
        flops = macs * 2
        
        # Run batch inference
        result = pipeline.batch_predict(num_images=1000)  # Process 50 images
        
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        model_time = time.time() - model_start_time
        
        print(f"{model_name:<20} {params/1e6:>6.1f}M {flops/1e9:>11.1f}G {result['avg_inference_time']:>13.3f}s {result['total_time']:>13.1f}s")
        
        # Print example predictions
        print(f"\nExample predictions for {model_name}:")
        for pred in result['predictions'][:3]:  # Show first 3 examples
            print(f"File: {pred['file']}")
            print(f"True: {pred['true_class']}, Predicted: {pred['top_pred']} ({pred['confidence']:.1f}%)")
        print(f"Overall accuracy: {result['accuracy']:.1f}%\n")
        
        timing_data.append({
            'model': model_name,
            'start_time': start_datetime,
            'end_time': end_datetime,
            'avg_inference_time': f"{result['avg_inference_time']:.3f}",
            'total_time': f"{result['total_time']:.1f}",
            'parameters': f"{params/1e6:.1f}",
            'flops': f"{flops/1e9:.1f}",
            'accuracy': f"{result['accuracy']:.1f}"
        })

    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    # Write timing data to CSV
    with open('inference_timing.csv', 'w', newline='') as f:
        fieldnames = ['model', 'start_time', 'end_time', 'avg_inference_time', 'total_time', 'parameters', 'flops', 'accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_data)

if __name__ == '__main__':
    main()