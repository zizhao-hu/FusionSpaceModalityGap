import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from collections import defaultdict
import re
from scipy.ndimage import gaussian_filter1d

def extract_generation_number(folder_path):
    """Extract generation number from folder path."""
    match = re.search(r'gen_(\d+)', folder_path)
    if match:
        return int(match.group(1))
    return None

def analyze_color_distribution(image_path):
    """Analyze the color distribution of an image."""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Extract RGB channels
        r_channel = img_array[:, :, 0].flatten()
        g_channel = img_array[:, :, 1].flatten()
        b_channel = img_array[:, :, 2].flatten()
        
        # Calculate histograms for each channel with larger bin size (5)
        num_bins = 51  # 256/5 ≈ 51 bins
        bin_edges = np.arange(0, 257, 5)  # Bin edges from 0 to 256 with step 5
        
        r_hist, _ = np.histogram(r_channel, bins=bin_edges)
        g_hist, _ = np.histogram(g_channel, bins=bin_edges)
        b_hist, _ = np.histogram(b_channel, bins=bin_edges)
        
        # Normalize histograms
        r_hist = r_hist / r_hist.sum()
        g_hist = g_hist / g_hist.sum()
        b_hist = b_hist / b_hist.sum()
        
        # Calculate average RGB values
        avg_r = np.mean(r_channel)
        avg_g = np.mean(g_channel)
        avg_b = np.mean(b_channel)
        
        # Calculate color saturation (difference between max and min channel values)
        saturation = np.mean(np.max(img_array, axis=2) - np.min(img_array, axis=2))
        
        # Calculate color variance (measure of color diversity)
        r_var = np.var(r_channel)
        g_var = np.var(g_channel)
        b_var = np.var(b_channel)
        color_variance = (r_var + g_var + b_var) / 3
        
        return {
            'r_hist': r_hist,
            'g_hist': g_hist,
            'b_hist': b_hist,
            'avg_rgb': (avg_r, avg_g, avg_b),
            'saturation': saturation,
            'color_variance': color_variance
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def collect_data_for_generations(base_dir, target_generations=[0, 4, 10]):
    """Collect color distribution data for specified generations."""
    # Dictionary to store results by generation
    results = {gen: [] for gen in target_generations}
    
    # Find all generation folders
    all_folders = glob.glob(os.path.join(base_dir, "sd_to_sd_*_gen_*"))
    
    # Group folders by generation
    gen_folders = defaultdict(list)
    for folder in all_folders:
        gen_num = extract_generation_number(folder)
        if gen_num is not None and gen_num in target_generations:
            gen_folders[gen_num].append(folder)
    
    # Process each generation
    for gen, folders in gen_folders.items():
        print(f"Processing generation {gen} ({len(folders)} folders)")
        
        for folder in folders:
            image_files = glob.glob(os.path.join(folder, "*.jpg"))
            
            for img_path in image_files:
                color_data = analyze_color_distribution(img_path)
                if color_data:
                    results[gen].append(color_data)
    
    return results

def plot_color_distributions(results, output_path):
    """Plot color distributions for different generations, with each subfigure showing one channel."""
    target_generations = sorted(results.keys())
    
    # Create figure with 3 subplots (one for each color channel) - make it wider for better readability in a paper
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # X-axis for histograms - adjusted for bin size of 5
    # Use bin centers instead of edges to match histogram length
    x = np.arange(2.5, 256, 5)  # Centers of bins (2.5, 7.5, 12.5, ..., 252.5)
    
    # Colors for each generation - make Gen 0 yellow and Gen 10 purple
    gen_colors = {}
    for gen in target_generations:
        if gen == 0:
            gen_colors[gen] = '#FFA500'  # Yellow-orange for Gen 0
        elif gen == 10:
            gen_colors[gen] = '#9370DB'  # Medium purple for Gen 10
        else:
            gen_colors[gen] = '#1f77b4'  # Default blue for other generations
    
    # Channel names, subplot titles, and plot colors
    channels = ['Red', 'Green', 'Blue']
    hist_keys = ['r_hist', 'g_hist', 'b_hist']
    channel_colors = ['#d62728', '#2ca02c', '#1f77b4']  # Red, Green, Blue
    
    # Find global y-max for consistent y-axis scaling across all subplots
    global_y_max = 0
    for hist_key in hist_keys:
        for gen in target_generations:
            gen_data = results[gen]
            if gen_data:
                avg_hist = np.mean([data[hist_key] for data in gen_data], axis=0)
                global_y_max = max(global_y_max, np.max(avg_hist))
    
    # Calculate channel-specific metrics for all generations
    channel_metrics = {}
    for i, hist_key in enumerate(hist_keys):
        channel_metrics[hist_key] = {}
        for gen in target_generations:
            gen_data = results[gen]
            if gen_data:
                # Mean value for this channel
                if hist_key == 'r_hist':
                    avg_val = np.mean([data['avg_rgb'][0] for data in gen_data])
                    # Calculate standard deviation instead of variance
                    std_val = np.sqrt(np.mean([data['color_variance'] for data in gen_data]) / 3)  # R component std
                elif hist_key == 'g_hist':
                    avg_val = np.mean([data['avg_rgb'][1] for data in gen_data])
                    std_val = np.sqrt(np.mean([data['color_variance'] for data in gen_data]) / 3)  # G component std
                else:  # b_hist
                    avg_val = np.mean([data['avg_rgb'][2] for data in gen_data])
                    std_val = np.sqrt(np.mean([data['color_variance'] for data in gen_data]) / 3)  # B component std
                
                # Saturation (same for all channels)
                sat_vals = [data['saturation'] for data in gen_data if 'saturation' in data]
                sat_val = np.mean(sat_vals) if sat_vals else 0
                
                channel_metrics[hist_key][gen] = {
                    'mean': avg_val,
                    'std': std_val,
                    'saturation': sat_val
                }
    
    # Plot each channel in a separate subplot
    for i, (channel, hist_key, channel_color) in enumerate(zip(channels, hist_keys, channel_colors)):
        ax = axes[i]
        
        # Store average values for annotation
        avg_values = {}
        
        # Plot each generation's distribution for this channel
        for gen in target_generations:
            gen_data = results[gen]
            
            if not gen_data:
                continue
            
            # Average the histogram for this channel across all images in this generation
            avg_hist = np.mean([data[hist_key] for data in gen_data], axis=0)
            
            # Apply a small amount of smoothing for better visualization
            smoothed_hist = gaussian_filter1d(avg_hist, sigma=1.0)
            
            # Plot the histogram for this generation
            ax.plot(x, smoothed_hist, color=gen_colors[gen], alpha=0.8, linewidth=4.0, 
                   label=f'Gen {gen}')
            
            # Get average value for this channel
            avg_val = channel_metrics[hist_key][gen]['mean']
            avg_values[gen] = avg_val
            
            # Add a vertical line at the average value
            ax.axvline(x=avg_val, color=gen_colors[gen], linestyle='--', alpha=0.7, linewidth=3.0)
        
        # Add a light background color matching the channel
        ax.set_facecolor(f"{channel_color}10")  # Very light version of the channel color
        
        # Set title with larger font sizes
        ax.set_title(f'{channel} Channel', fontsize=22, fontweight='bold', color=channel_color)
        
        # Only show x-axis title for middle figure
        if i == 1:
            ax.set_xlabel('Pixel Intensity (bin size = 5)', fontsize=20)
        else:
            ax.set_xlabel('')
        
        # Only show y-axis title for first figure
        if i == 0:
            ax.set_ylabel('Frequency (log scale)', fontsize=20)
        else:
            ax.set_ylabel('')
        
        # Add legend only to the first figure
        if i == 0:
            ax.legend(loc='upper left', fontsize=18)
        
        # Add grid lines for better readability
        ax.grid(True, alpha=0.4, linestyle='--')
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # Set y-axis limits to 10^-2 to 10^-1 and use log scale
        ax.set_ylim(0.01, 0.1)  # From 10^-2 to 10^-1
        ax.set_yscale('log')  # Use logarithmic scale for frequency
        
        # Remove intermediate tick values between 10^-2 and 10^-1
        ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=2))  # Only show 10^-2 and 10^-1
        
        # Add shift annotation in the center of each figure
        if 0 in avg_values and 10 in avg_values:
            # Calculate the shift from gen 0 to gen 10
            shift = avg_values[10] - avg_values[0]
            
            # Add shift text in the center of the plot
            ax.text(0.5, 0.5, f"Shift: {shift:+.1f}",
                   transform=ax.transAxes,
                   fontsize=18, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add metrics summary only in the first figure
            if i == 0:
                # Create a summary text with all metrics in the first figure (without shift)
                metrics_text = f"Mean @0: {avg_values[0]:.1f}\n"
                metrics_text += f"Mean @10: {avg_values[10]:.1f}\n"
                metrics_text += f"Std @0: {channel_metrics[hist_key][0]['std']:.1f}\n"
                metrics_text += f"Std @10: {channel_metrics[hist_key][10]['std']:.1f}\n"
                metrics_text += f"Saturation @0: {channel_metrics[hist_key][0]['saturation']:.1f}\n"
                metrics_text += f"Saturation @10: {channel_metrics[hist_key][10]['saturation']:.1f}"
                
                # Add text box in top right corner
                ax.text(0.98, 0.98, metrics_text,
                       transform=ax.transAxes,
                       fontsize=14, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

def create_color_swatch_plot(results, output_path):
    """Create a plot showing the average color of each generation."""
    target_generations = sorted(results.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate average RGB for each generation
    avg_colors = []
    labels = []
    
    for gen in target_generations:
        gen_data = results[gen]
        if gen_data:
            # Average RGB across all images in this generation
            avg_rgb = np.mean([data['avg_rgb'] for data in gen_data], axis=0)
            avg_colors.append(avg_rgb / 255.0)  # Normalize to 0-1 for matplotlib
            labels.append(f"Gen {gen}")
    
    # Plot color swatches
    for i, (color, label) in enumerate(zip(avg_colors, labels)):
        ax.add_patch(plt.Rectangle((i, 0), 0.8, 1, color=color))
        
        # Add text with RGB values - larger font
        rgb_text = f"{label}\nR: {color[0]*255:.1f}\nG: {color[1]*255:.1f}\nB: {color[2]*255:.1f}"
        ax.text(i + 0.4, 0.5, rgb_text, ha='center', va='center', 
                fontsize=16, fontweight='bold',
                color='white' if np.mean(color) < 0.5 else 'black')
    
    # Set axis properties
    ax.set_xlim(0, len(avg_colors))
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Color swatch plot saved to {output_path}")
    plt.close()

def plot_metrics_trend(results, output_path):
    """Plot the trend of color metrics across generations."""
    target_generations = sorted(results.keys())
    
    # Extract metrics for each generation
    generations = []
    avg_r_values = []
    avg_g_values = []
    avg_b_values = []
    saturation_values = []
    variance_values = []
    
    for gen in target_generations:
        gen_data = results[gen]
        if gen_data:
            generations.append(gen)
            
            # Calculate average metrics
            avg_rgb = np.mean([data['avg_rgb'] for data in gen_data], axis=0)
            avg_r_values.append(avg_rgb[0])
            avg_g_values.append(avg_rgb[1])
            avg_b_values.append(avg_rgb[2])
            
            saturation_values.append(np.mean([data['saturation'] for data in gen_data]))
            variance_values.append(np.mean([data['color_variance'] for data in gen_data]) / 100)  # Scale down for better visualization
    
    if not generations:
        print("No data available for metrics trend plot")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot RGB values
    ax1.plot(generations, avg_r_values, 'o-', color='red', linewidth=2.5, markersize=10, label='Red')
    ax1.plot(generations, avg_g_values, 's-', color='green', linewidth=2.5, markersize=10, label='Green')
    ax1.plot(generations, avg_b_values, '^-', color='blue', linewidth=2.5, markersize=10, label='Blue')
    
    # Add data labels - larger font
    for i, gen in enumerate(generations):
        ax1.annotate(f"{avg_r_values[i]:.1f}", (gen, avg_r_values[i]), 
                    xytext=(0, 10), textcoords='offset points', ha='center', fontsize=14, color='red')
        ax1.annotate(f"{avg_g_values[i]:.1f}", (gen, avg_g_values[i]), 
                    xytext=(0, -15), textcoords='offset points', ha='center', fontsize=14, color='green')
        ax1.annotate(f"{avg_b_values[i]:.1f}", (gen, avg_b_values[i]), 
                    xytext=(0, 10), textcoords='offset points', ha='center', fontsize=14, color='blue')
    
    ax1.set_title('Average RGB Values', fontsize=18)
    ax1.set_ylabel('Pixel Intensity', fontsize=16)
    ax1.legend(loc='upper left', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot saturation and variance
    ax2.plot(generations, saturation_values, 'o-', color='purple', linewidth=2.5, markersize=10, label='Saturation')
    ax2.plot(generations, variance_values, 's-', color='orange', linewidth=2.5, markersize=10, label='Variance/100')
    
    # Add data labels - larger font
    for i, gen in enumerate(generations):
        ax2.annotate(f"{saturation_values[i]:.1f}", (gen, saturation_values[i]), 
                    xytext=(0, 10), textcoords='offset points', ha='center', fontsize=14, color='purple')
        ax2.annotate(f"{variance_values[i]*100:.1f}", (gen, variance_values[i]), 
                    xytext=(0, -15), textcoords='offset points', ha='center', fontsize=14, color='orange')
    
    ax2.set_title('Color Saturation and Variance', fontsize=18)
    ax2.set_xlabel('Generation', fontsize=16)
    ax2.set_ylabel('Value', fontsize=16)
    ax2.legend(loc='upper left', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Set x-ticks to be exactly at the generation numbers
    plt.xticks(generations)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics trend plot saved to {output_path}")
    plt.close()

def main():
    # Base directory containing the image folders
    base_dir = os.path.join("data", "coco")
    
    # Target generations to analyze
    target_generations = [0, 4, 10]
    
    # Collect data
    print(f"Collecting color distribution data for generations {target_generations}...")
    results = collect_data_for_generations(base_dir, target_generations)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("data", "color_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the results
    plot_output_path = os.path.join(output_dir, "color_distribution.png")
    plot_color_distributions(results, plot_output_path)
    
    # Create color swatch plot
    swatch_output_path = os.path.join(output_dir, "color_swatches.png")
    create_color_swatch_plot(results, swatch_output_path)
    
    # Create metrics trend plot
    trend_output_path = os.path.join(output_dir, "color_metrics_trend.png")
    plot_metrics_trend(results, trend_output_path)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for gen in target_generations:
        gen_data = results[gen]
        if gen_data:
            # Ensure we have RGB values for all generations
            avg_rgb = np.mean([data['avg_rgb'] for data in gen_data], axis=0)
            
            # Handle saturation which might be missing in some data points
            saturation_values = [data['saturation'] for data in gen_data if 'saturation' in data]
            avg_saturation = np.mean(saturation_values) if saturation_values else 0
            
            avg_variance = np.mean([data['color_variance'] for data in gen_data])
            
            print(f"Generation {gen} ({len(gen_data)} images):")
            print(f"  Average RGB: ({avg_rgb[0]:.1f}, {avg_rgb[1]:.1f}, {avg_rgb[2]:.1f})")
            if avg_saturation > 0:
                print(f"  Average Saturation: {avg_saturation:.1f}")
            print(f"  Average Color Variance: {avg_variance:.1f}")
        else:
            print(f"Generation {gen}: No data")

if __name__ == "__main__":
    main() 