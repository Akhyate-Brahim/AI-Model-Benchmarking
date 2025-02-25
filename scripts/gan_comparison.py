import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import csv
import numpy as np

class DCGANInferencePipeline:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the pre-trained DCGAN model"""
        try:
            print(f"Loading DCGAN model on {'GPU' if self.device == 'cuda' else 'CPU'}...")
            model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 
                                   'DCGAN', 
                                   pretrained=True, 
                                   useGPU=(self.device == 'cuda'))
            return model
        except Exception as e:
            print(f"Error loading DCGAN model: {e}")
            raise e
    
    def generate_images(self, num_images=10, seed=None):
        """Generate images using the DCGAN model"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        start_time = time.time()
        
        # Build noise data according to the model's specifications
        noise, _ = self.model.buildNoiseData(num_images)
        
        # Generate images
        with torch.no_grad():
            generated_images = self.model.test(noise)
        
        inference_time = time.time() - start_time
        
        return {
            'images': generated_images.detach().cpu(),
            'inference_time': inference_time,
            'avg_time_per_image': inference_time / num_images
        }
    
    def save_images(self, images, output_dir='generated_images'):
        """Save generated images to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a grid of images
        grid = vutils.make_grid(images, padding=2, normalize=True)
        
        # Save the grid
        grid_path = os.path.join(output_dir, f"dcgan_{timestamp}_grid.png")
        vutils.save_image(grid, grid_path)
        
        # Save individual images
        for i, image in enumerate(images):
            img_path = os.path.join(output_dir, f"dcgan_{timestamp}_{i}.png")
            vutils.save_image(image, img_path)
        
        return grid_path
    
    def benchmark(self, batch_sizes=[1, 4, 8, 16, 32, 64], num_runs=3):
        """Benchmark the model with different batch sizes"""
        results = []
        
        for batch_size in batch_sizes:
            batch_times = []
            for run in range(num_runs):
                try:
                    # Use a different seed for each run
                    seed = 42 + run
                    result = self.generate_images(num_images=batch_size, seed=seed)
                    batch_times.append(result['inference_time'])
                except RuntimeError as e:
                    # Handle out of memory errors gracefully
                    if 'out of memory' in str(e).lower():
                        print(f"Batch size {batch_size} too large for GPU memory. Skipping.")
                        break
                    else:
                        raise e
            
            # Only calculate statistics if we have results
            if batch_times:
                avg_time = np.mean(batch_times)
                throughput = batch_size / avg_time  # images per second
                
                results.append({
                    'batch_size': batch_size,
                    'avg_time': avg_time,
                    'throughput': throughput,
                    'images_per_second': throughput
                })
            
        return results

def main():
    total_start_time = time.time()
    timing_data = []
    
    # Device selection - use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Batch sizes to test for benchmarking
    batch_sizes = [1, 4, 8, 16, 32, 64] if device == 'cuda' else [1, 2, 4, 8]
    
    try:
        # Initialize the DCGAN pipeline
        pipeline = DCGANInferencePipeline(device=device)
        
        print("\nDCGAN Benchmark:")
        print("-" * 80)
        print(f"{'Batch Size':<15} {'Time per Batch(s)':<20} {'Images/Second':<15}")
        print("-" * 80)
        
        # Run benchmarks
        benchmark_results = pipeline.benchmark(batch_sizes=batch_sizes)
        
        # Generate sample images
        generation_result = pipeline.generate_images(num_images=16, seed=42)
        
        # Save generated images
        grid_path = pipeline.save_images(generation_result['images'])
        
        # Print benchmark results
        for result in benchmark_results:
            print(f"{result['batch_size']:<15} {result['avg_time']:>15.3f}s {result['images_per_second']:>12.2f}")
        
        # Display one example grid using matplotlib
        grid = vutils.make_grid(generation_result['images'], padding=2, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.savefig("dcgan_preview.png")
        print(f"\nSample images saved to: {grid_path}")
        print(f"Preview image saved to: dcgan_preview.png")
        
        # Gather timing data for CSV
        for result in benchmark_results:
            timing_data.append({
                'model': 'DCGAN',
                'batch_size': result['batch_size'],
                'avg_time_per_batch': f"{result['avg_time']:.3f}",
                'images_per_second': f"{result['images_per_second']:.2f}"
            })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Write timing data to CSV
    if timing_data:
        with open('dcgan_inference_timing.csv', 'w', newline='') as f:
            fieldnames = ['model', 'batch_size', 'avg_time_per_batch', 'images_per_second']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(timing_data)
    
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()