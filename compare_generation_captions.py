import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import textwrap
import re
from PIL import Image

def load_captions(gen_number):
    """Load captions from a specific generation."""
    file_path = f"data/vlm_captions/gen_{gen_number}/all_captions.json"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    captions = [item["generated_caption"] for item in data]
    image_paths = [item["image_path"] for item in data]
    
    return captions, image_paths

def calculate_caption_stats(captions):
    """Calculate statistics for a list of captions."""
    if not captions:
        return {
            "count": 0,
            "avg_length": 0,
            "avg_word_count": 0,
            "min_length": 0,
            "max_length": 0,
            "min_word_count": 0,
            "max_word_count": 0
        }
    
    lengths = [len(caption) for caption in captions]
    word_counts = [len(caption.split()) for caption in captions]
    
    return {
        "count": len(captions),
        "avg_length": sum(lengths) / len(captions),
        "avg_word_count": sum(word_counts) / len(captions),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "min_word_count": min(word_counts),
        "max_word_count": max(word_counts)
    }

def clean_caption(caption):
    """Remove common prefixes from captions."""
    prefixes = [
        "a detailed photo of ",
        "a photo with many details showing ",
        "a detailed scene showing ",
        "a photo of ",
        "a picture of ",
        "an image of "
    ]
    
    for prefix in prefixes:
        if caption.lower().startswith(prefix):
            return caption[len(prefix):]
    
    return caption

def create_combined_visualization(gen_numbers, stats_list, captions_list, image_path, output_path):
    """Create a combined visualization with statistics and caption comparison."""
    # Clean captions by removing common prefixes
    cleaned_captions = [clean_caption(caption) for caption in captions_list]
    
    # Create figure with custom layout - more compact
    fig = plt.figure(figsize=(12, 7))
    
    if image_path and os.path.exists(image_path):
        # Include the image in the layout
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 0.05, 0.6], width_ratios=[1, 1, 1])
        
        # Add the image
        ax_img = fig.add_subplot(gs[0, 2])
        img = Image.open(image_path)
        ax_img.imshow(img)
        ax_img.set_title("Ground Truth Image", fontsize=10, fontweight='bold')
        ax_img.axis('off')
    else:
        # No image, use original layout
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 0.05, 0.6])
        print(f"Warning: Could not load image at {image_path}")
    
    # Plot average character length
    ax1 = fig.add_subplot(gs[0, 0])
    avg_lengths = [stats["avg_length"] for stats in stats_list]
    bars = ax1.bar([f"Gen {gen}" for gen in gen_numbers], avg_lengths, color=['blue', 'green', 'red'])
    ax1.set_title("Average Caption Length (Characters)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Characters", fontsize=9)
    
    # Set y-axis limit with some padding to prevent overflow
    max_length = max(avg_lengths) * 1.15  # Add 15% padding
    ax1.set_ylim(0, max_length)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot average word count
    ax2 = fig.add_subplot(gs[0, 1])
    avg_word_counts = [stats["avg_word_count"] for stats in stats_list]
    bars = ax2.bar([f"Gen {gen}" for gen in gen_numbers], avg_word_counts, color=['blue', 'green', 'red'])
    ax2.set_title("Average Caption Word Count", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Words", fontsize=9)
    
    # Set y-axis limit with some padding to prevent overflow
    max_words = max(avg_word_counts) * 1.15  # Add 15% padding
    ax2.set_ylim(0, max_words)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Create a single subplot for all captions - very compact
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis('off')
    
    # Display all captions in a single text box with minimal spacing
    caption_text = f"Gen {gen_numbers[0]}: {cleaned_captions[0]}\n" + \
                  f"Gen {gen_numbers[1]}: {cleaned_captions[1]}\n" + \
                  f"Gen {gen_numbers[2]}: {cleaned_captions[2]}"
    
    ax3.text(0.01, 0.99, caption_text, fontsize=9, 
             verticalalignment='top', horizontalalignment='left',
             transform=ax3.transAxes,
             bbox=dict(facecolor='white', alpha=0.9, pad=3, boxstyle='round,pad=0.3'))
    
    # Adjust layout to be more compact
    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create HTML comparison for better text display
    create_html_comparison(cleaned_captions[0], cleaned_captions[1], cleaned_captions[2], 
                          "data/visualizations/caption_comparison_last.html", image_path)

def create_html_comparison(caption_gen0, caption_gen4, caption_gen10, output_path, image_path=None):
    """Create an HTML file for caption comparison."""
    image_html = ""
    if image_path and os.path.exists(image_path):
        # Convert the image path to a relative path for the HTML
        rel_path = os.path.relpath(image_path, os.path.dirname(output_path))
        image_html = f"""
        <div class="image-container">
            <h2>Ground Truth Image</h2>
            <img src="{rel_path}" alt="Ground Truth Image" style="max-width: 300px; max-height: 300px;">
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Caption Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 800px; margin: 0 auto; }}
            .caption-container {{ margin-bottom: 5px; padding: 5px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
            .caption-title {{ font-weight: bold; margin-bottom: 0; font-size: 12px; }}
            .caption-text {{ line-height: 1.2; font-size: 12px; margin-top: 0; }}
            .image-container {{ text-align: center; margin-bottom: 15px; }}
        </style>
    </head>
    <body>
        <h1>Caption Comparison Across Generations</h1>
        
        {image_html}
        
        <div class="caption-container">
            <div class="caption-title">Generation 0:</div>
            <div class="caption-text">{caption_gen0}</div>
        </div>
        
        <div class="caption-container">
            <div class="caption-title">Generation 4:</div>
            <div class="caption-text">{caption_gen4}</div>
        </div>
        
        <div class="caption-container">
            <div class="caption-title">Generation 10:</div>
            <div class="caption-text">{caption_gen10}</div>
        </div>
    </body>
    </html>
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML comparison saved to {output_path}")

def main():
    # Load captions for each generation
    gen_numbers = [0, 4, 10]
    captions_lists = []
    image_paths_lists = []
    stats_list = []
    
    for gen in gen_numbers:
        captions, image_paths = load_captions(gen)
        captions_lists.append(captions)
        image_paths_lists.append(image_paths)
        
        stats = calculate_caption_stats(captions)
        stats_list.append(stats)
        
        print(f"\nGeneration {gen} Caption Statistics:")
        print(f"  Count: {stats['count']}")
        print(f"  Average Length: {stats['avg_length']:.2f} characters")
        print(f"  Average Word Count: {stats['avg_word_count']:.2f} words")
        print(f"  Length Range: {stats['min_length']} - {stats['max_length']} characters")
        print(f"  Word Count Range: {stats['min_word_count']} - {stats['max_word_count']} words")
    
    # Use the last caption example (index -1 or 99 if available)
    caption_index = min(99, len(captions_lists[0]) - 1)
    
    # Get the image path from the first generation's data
    image_path = image_paths_lists[0][caption_index] if image_paths_lists[0] else None
    
    # Create combined visualization
    create_combined_visualization(
        gen_numbers,
        stats_list,
        [captions_lists[0][caption_index], captions_lists[1][caption_index], captions_lists[2][caption_index]],
        image_path,
        "data/visualizations/caption_comparison_combined.png"
    )
    
    # Print the last caption from each generation for comparison
    print(f"\nLast Caption Comparison (Index {caption_index}):")
    print(f"  Gen 0: \"{captions_lists[0][caption_index]}\"")
    print(f"  Gen 4: \"{captions_lists[1][caption_index]}\"")
    print(f"  Gen 10: \"{captions_lists[2][caption_index]}\"")
    
    # Also print cleaned versions
    print("\nCleaned Last Caption Comparison:")
    print(f"  Gen 0: \"{clean_caption(captions_lists[0][caption_index])}\"")
    print(f"  Gen 4: \"{clean_caption(captions_lists[1][caption_index])}\"")
    print(f"  Gen 10: \"{clean_caption(captions_lists[2][caption_index])}\"")

if __name__ == "__main__":
    main() 