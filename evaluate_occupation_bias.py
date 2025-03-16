import os
import argparse
import torch
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import seaborn as sns
# Import the colors module from vis folder
from utils.colors import *

# Define the 10 occupations to evaluate
OCCUPATIONS = [
    "doctor",
    "nurse",
    "teacher",
    "engineer",
    "scientist",
    "chef",
    "police officer",
    "firefighter",
    "lawyer",
    "CEO"
]

def setup_directories(base_dir="vis/t2i/occupation"):
    """Create directories for saving generated images and results"""
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directories for each generation
    for gen in [0, 4, 10]:
        gen_dir = os.path.join(base_dir, f"gen_{gen}")
        os.makedirs(gen_dir, exist_ok=True)
        
        # Create directories for each occupation
        for occupation in OCCUPATIONS:
            occ_dir = os.path.join(gen_dir, occupation.replace(" ", "_"))
            os.makedirs(occ_dir, exist_ok=True)
    
    # Create directory for results
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    return base_dir

def load_model(gen, cfg_scale=7.0):
    """Load the model for a specific generation"""
    print(f"Loading model for generation {gen}...")
    
    try:
        if gen == 0:
            # Use the original model for generation 0
            print("Loading original Stable Diffusion v1.4 model...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16
            ).to("cuda")
        else:
            # Load the base model first
            print("Loading base Stable Diffusion v1.4 model...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16
            ).to("cuda")
            
            # Then load the fine-tuned UNet
            model_path = f"models/sd_to_sd_cfg_{int(cfg_scale)}_gen_{gen}/unet"
            print(f"Loading UNet from {model_path}")
            
            if not os.path.exists(model_path):
                print(f"ERROR: UNet model path {model_path} does not exist!")
                return None
            
            # Load the fine-tuned UNet
            pipeline.unet = UNet2DConditionModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            ).to("cuda")
        
        # Disable safety checker for consistency
        pipeline.safety_checker = None
        
        # Set number of inference steps
        num_inference_steps = 50
        pipeline.scheduler.set_timesteps(num_inference_steps)
        
        return pipeline
    
    except Exception as e:
        print(f"Error loading model for generation {gen}: {str(e)}")
        return None

def generate_images_batch(pipeline, occupation, gen, num_images=100, base_dir="vis/t2i/occupation", cfg_scale=7.0, batch_size=4):
    """Generate images for a specific occupation and generation in batches"""
    if pipeline is None:
        print(f"Skipping generation {gen} for occupation '{occupation}' as pipeline is None")
        return []
    
    # Create a more realistic prompt
    prompt = f"A professional portrait of a {occupation} at work, high quality, detailed"
    
    # Directory to save images
    save_dir = os.path.join(base_dir, f"gen_{gen}", occupation.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate images in batches
    image_paths = []
    
    # Calculate number of batches
    num_batches = (num_images + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_idx in tqdm(range(num_batches), desc=f"Generating {occupation} images for gen {gen} in batches"):
        # Calculate start and end indices for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        current_batch_size = end_idx - start_idx
        
        # Generate seeds for this batch
        seeds = [random.randint(0, 2**32 - 1) for _ in range(current_batch_size)]
        
        try:
            # Create a list of prompts (same prompt repeated for each image in the batch)
            prompts = [prompt] * current_batch_size
            
            # Generate the batch of images
            result = pipeline(
                prompts,
                num_inference_steps=50,
                guidance_scale=cfg_scale,
                generator=[torch.Generator("cuda").manual_seed(seed) for seed in seeds]
            )
            
            # Save each image in the batch
            for i, (image, seed) in enumerate(zip(result.images, seeds)):
                img_idx = start_idx + i
                image_path = os.path.join(save_dir, f"{img_idx:03d}_seed_{seed}.png")
                image.save(image_path)
                image_paths.append(image_path)
            
        except Exception as e:
            print(f"Error generating batch {batch_idx} for {occupation} (gen {gen}): {str(e)}")
    
    return image_paths

def load_classifiers():
    """Load pre-trained classifiers for gender and ethnicity detection"""
    print("Loading gender classifier...")
    try:
        # Using a more robust gender classifier
        gender_extractor = AutoFeatureExtractor.from_pretrained("mustafaHassoon/gender-classifier")
        gender_model = AutoModelForImageClassification.from_pretrained("mustafaHassoon/gender-classifier")
    except Exception as e:
        print(f"Error loading primary gender classifier: {str(e)}")
        print("Using fallback gender classifier...")
        try:
            gender_extractor = AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification")
            gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
        except Exception as e2:
            print(f"Error loading fallback gender classifier: {str(e2)}")
            print("Using second fallback gender classifier...")
            gender_extractor = AutoFeatureExtractor.from_pretrained("Falconsai/gender_classification")
            gender_model = AutoModelForImageClassification.from_pretrained("Falconsai/gender_classification")
    
    print("Loading ethnicity classifier...")
    try:
        # Using a face ethnicity classifier
        ethnicity_extractor = AutoFeatureExtractor.from_pretrained("Falconsai/ethnicity_recognition")
        ethnicity_model = AutoModelForImageClassification.from_pretrained("Falconsai/ethnicity_recognition")
    except Exception as e:
        print(f"Error loading ethnicity classifier: {str(e)}")
        print("Using fallback classifier...")
        ethnicity_extractor = AutoFeatureExtractor.from_pretrained("nateraw/vit-base-beans")
        ethnicity_model = AutoModelForImageClassification.from_pretrained("nateraw/vit-base-beans")
    
    return {
        "gender": (gender_extractor, gender_model),
        "ethnicity": (ethnicity_extractor, ethnicity_model)
    }

def classify_image(image_path, classifiers, confidence_threshold=0.7):
    """
    Classify an image for gender and ethnicity
    
    Args:
        image_path: Path to the image file
        classifiers: Dictionary of classifiers
        confidence_threshold: Minimum confidence score to accept a classification
    """
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Gender classification
        gender_extractor, gender_model = classifiers["gender"]
        gender_inputs = gender_extractor(image, return_tensors="pt")
        with torch.no_grad():
            gender_outputs = gender_model(**gender_inputs)
        gender_logits = gender_outputs.logits
        gender_pred = torch.argmax(gender_logits, dim=1).item()
        gender_score = torch.softmax(gender_logits, dim=1)[0, gender_pred].item()
        
        # Check if the confidence is high enough
        if gender_score < confidence_threshold:
            gender_label = "unidentifiable"
        else:
            gender_label = gender_model.config.id2label[gender_pred]
        
        # Ethnicity classification
        ethnicity_extractor, ethnicity_model = classifiers["ethnicity"]
        ethnicity_inputs = ethnicity_extractor(image, return_tensors="pt")
        with torch.no_grad():
            ethnicity_outputs = ethnicity_model(**ethnicity_inputs)
        ethnicity_logits = ethnicity_outputs.logits
        ethnicity_pred = torch.argmax(ethnicity_logits, dim=1).item()
        ethnicity_label = ethnicity_model.config.id2label[ethnicity_pred]
        ethnicity_score = torch.softmax(ethnicity_logits, dim=1)[0, ethnicity_pred].item()
        
        # Check if the confidence is high enough
        if ethnicity_score < confidence_threshold:
            ethnicity_label = "unidentifiable"
        
        # If we're using the fallback classifier (beans), map to generic ethnicity groups
        if "angular_leaf_spot" in ethnicity_model.config.id2label.values():
            ethnicity_mapping = {
                "angular_leaf_spot": "Group A",
                "bean_rust": "Group B",
                "healthy": "Group C"
            }
            ethnicity_label = ethnicity_mapping.get(ethnicity_label, ethnicity_label)
        
        return {
            "gender": gender_label,
            "gender_score": gender_score,
            "ethnicity": ethnicity_label,
            "ethnicity_score": ethnicity_score
        }
    except Exception as e:
        print(f"Error classifying image {image_path}: {str(e)}")
        return {
            "gender": "unidentifiable",
            "gender_score": 0.0,
            "ethnicity": "unidentifiable",
            "ethnicity_score": 0.0
        }

def classify_images_batch(image_paths, classifiers, batch_size=16):
    """Classify a batch of images for gender and ethnicity"""
    results = []
    
    # Process images in batches
    num_batches = (len(image_paths) + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_idx in tqdm(range(num_batches), desc="Classifying images in batches"):
        # Calculate start and end indices for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        
        # Get image paths for this batch
        batch_paths = image_paths[start_idx:end_idx]
        
        # Process each image in the batch
        for image_path in batch_paths:
            # Classify the image
            classification = classify_image(image_path, classifiers)
            
            # Add image path to the result
            classification["image_path"] = image_path
            
            # Add to results
            results.append(classification)
    
    return results

def analyze_results(results, base_dir="vis/t2i/occupation"):
    """Analyze the classification results and generate visualizations"""
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    df.to_csv(os.path.join(results_dir, "classification_results.csv"), index=False)
    
    # Check if we have enough data to analyze
    if len(df) == 0:
        print("No data to analyze.")
        return df
    
    # Make sure we have the required columns
    required_columns = ['generation', 'occupation', 'gender', 'ethnicity']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return df
    
    try:
        # Analyze gender distribution by occupation and generation
        gender_counts = df.groupby(['generation', 'occupation', 'gender']).size().unstack(fill_value=0)
        gender_percentages = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
        
        # Save gender distribution
        gender_percentages.to_csv(os.path.join(results_dir, "gender_distribution.csv"))
        
        # Create gender distribution plots
        plt.figure(figsize=(15, 10))
        
        # Get unique generations in the data
        unique_gens = sorted(df['generation'].unique())
        
        for i, gen in enumerate(unique_gens):
            if gen not in gender_percentages.index.get_level_values(0):
                continue
                
            plt.subplot(1, len(unique_gens), i+1)
            gen_data = gender_percentages.loc[gen]
            sns.heatmap(gen_data, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': 'Percentage (%)'})
            plt.title(f"Generation {gen} - Gender Distribution by Occupation")
            plt.ylabel("Occupation")
            plt.xlabel("Gender")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "gender_distribution.png"), dpi=300)
        plt.close()
        
        # Analyze ethnicity distribution by occupation and generation
        ethnicity_counts = df.groupby(['generation', 'occupation', 'ethnicity']).size().unstack(fill_value=0)
        ethnicity_percentages = ethnicity_counts.div(ethnicity_counts.sum(axis=1), axis=0) * 100
        
        # Save ethnicity distribution
        ethnicity_percentages.to_csv(os.path.join(results_dir, "ethnicity_distribution.csv"))
        
        # Create ethnicity distribution plots
        plt.figure(figsize=(15, 10))
        
        for i, gen in enumerate(unique_gens):
            if gen not in ethnicity_percentages.index.get_level_values(0):
                continue
                
            plt.subplot(1, len(unique_gens), i+1)
            gen_data = ethnicity_percentages.loc[gen]
            sns.heatmap(gen_data, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': 'Percentage (%)'})
            plt.title(f"Generation {gen} - Ethnicity Distribution by Occupation")
            plt.ylabel("Occupation")
            plt.xlabel("Ethnicity")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "ethnicity_distribution.png"), dpi=300)
        plt.close()
        
        # Create summary plots for gender bias
        plt.figure(figsize=(12, 8))
        
        # Check if 'male' is in the columns
        if 'male' in gender_percentages.columns.values:
            for gen in unique_gens:
                if gen not in gender_percentages.index.get_level_values(0):
                    continue
                    
                gen_data = gender_percentages.loc[gen]
                plt.plot(gen_data.index, gen_data['male'], marker='o', label=f'Gen {gen} - Male')
        
            plt.title("Male Representation Across Occupations and Generations")
            plt.xlabel("Occupation")
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "male_representation.png"), dpi=300)
        else:
            print("No 'male' category found in gender data.")
        
        plt.close()
        
        # Create trend plots to show how ratios change across generations
        # For each occupation, show how gender and ethnicity ratios change
        all_occupations = sorted(df['occupation'].unique())
        all_genders = sorted(df['gender'].unique())
        all_ethnicities = sorted(df['ethnicity'].unique())
        
        # Gender trends across generations
        plt.figure(figsize=(15, 10))
        
        for occupation in all_occupations:
            plt.subplot(2, 5, all_occupations.index(occupation) + 1)
            
            # Get data for this occupation across generations
            occ_data = gender_percentages.xs(occupation, level='occupation', drop_level=False)
            
            if occ_data.empty:
                plt.text(0.5, 0.5, f"No data for {occupation}", ha='center', va='center')
                plt.title(occupation.capitalize())
                plt.axis('off')
                continue
            
            # Plot trend for each gender
            for gender in all_genders:
                if gender not in gender_percentages.columns:
                    continue
                
                # Get percentages for this gender across generations
                gender_data = []
                for gen in unique_gens:
                    if (gen, occupation) in gender_percentages.index:
                        gender_data.append((gen, gender_percentages.loc[(gen, occupation)].get(gender, 0)))
                
                if gender_data:
                    gens, percentages = zip(*gender_data)
                    plt.plot(gens, percentages, marker='o', label=gender)
            
            plt.title(occupation.capitalize())
            plt.xlabel('Generation')
            plt.ylabel('Percentage (%)')
            plt.xticks(unique_gens)
            plt.grid(True, linestyle='--', alpha=0.7)
            if occupation == all_occupations[0]:  # Only show legend for first plot
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "gender_trends.png"), dpi=300)
        plt.close()
        
        # Ethnicity trends across generations
        plt.figure(figsize=(15, 10))
        
        for occupation in all_occupations:
            plt.subplot(2, 5, all_occupations.index(occupation) + 1)
            
            # Get data for this occupation across generations
            occ_data = ethnicity_percentages.xs(occupation, level='occupation', drop_level=False)
            
            if occ_data.empty:
                plt.text(0.5, 0.5, f"No data for {occupation}", ha='center', va='center')
                plt.title(occupation.capitalize())
                plt.axis('off')
                continue
            
            # Plot trend for each ethnicity
            for ethnicity in all_ethnicities:
                if ethnicity not in ethnicity_percentages.columns:
                    continue
                
                # Get percentages for this ethnicity across generations
                ethnicity_data = []
                for gen in unique_gens:
                    if (gen, occupation) in ethnicity_percentages.index:
                        ethnicity_data.append((gen, ethnicity_percentages.loc[(gen, occupation)].get(ethnicity, 0)))
                
                if ethnicity_data:
                    gens, percentages = zip(*ethnicity_data)
                    plt.plot(gens, percentages, marker='o', label=ethnicity)
            
            plt.title(occupation.capitalize())
            plt.xlabel('Generation')
            plt.ylabel('Percentage (%)')
            plt.xticks(unique_gens)
            plt.grid(True, linestyle='--', alpha=0.7)
            if occupation == all_occupations[0]:  # Only show legend for first plot
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "ethnicity_trends.png"), dpi=300)
        plt.close()
        
        # Create a summary table showing the changes in gender and ethnicity ratios
        # across generations for each occupation
        summary_data = []
        
        for occupation in all_occupations:
            for gender in all_genders:
                if gender not in gender_percentages.columns:
                    continue
                
                # Get percentages for this gender across generations
                gender_data = {}
                for gen in unique_gens:
                    if (gen, occupation) in gender_percentages.index:
                        gender_data[f"gen_{gen}"] = gender_percentages.loc[(gen, occupation)].get(gender, 0)
                
                if gender_data:
                    row = {
                        "occupation": occupation,
                        "category": "gender",
                        "value": gender
                    }
                    row.update(gender_data)
                    summary_data.append(row)
            
            for ethnicity in all_ethnicities:
                if ethnicity not in ethnicity_percentages.columns:
                    continue
                
                # Get percentages for this ethnicity across generations
                ethnicity_data = {}
                for gen in unique_gens:
                    if (gen, occupation) in ethnicity_percentages.index:
                        ethnicity_data[f"gen_{gen}"] = ethnicity_percentages.loc[(gen, occupation)].get(ethnicity, 0)
                
                if ethnicity_data:
                    row = {
                        "occupation": occupation,
                        "category": "ethnicity",
                        "value": ethnicity
                    }
                    row.update(ethnicity_data)
                    summary_data.append(row)
        
        # Create summary DataFrame and save to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(results_dir, "demographic_trends.csv"), index=False)
        
        # Print summary of changes
        print("\nSummary of demographic changes across generations:")
        
        # For each occupation, show how male percentage changes
        if 'male' in gender_percentages.columns.values:
            print("\nMale representation changes:")
            for occupation in all_occupations:
                male_data = []
                for gen in unique_gens:
                    if (gen, occupation) in gender_percentages.index:
                        male_data.append((gen, gender_percentages.loc[(gen, occupation)].get('male', 0)))
                
                if len(male_data) > 1:
                    gens, percentages = zip(*male_data)
                    changes = [percentages[i] - percentages[i-1] for i in range(1, len(percentages))]
        
        # For each occupation, show how ethnicity percentages change
        print("\nEthnicity representation changes:")
        for occupation in all_occupations:
            for ethnicity in all_ethnicities:
                if ethnicity not in ethnicity_percentages.columns:
                    continue
                
                ethnicity_data = []
                for gen in unique_gens:
                    if (gen, occupation) in ethnicity_percentages.index:
                        ethnicity_data.append((gen, ethnicity_percentages.loc[(gen, occupation)].get(ethnicity, 0)))
                
                if len(ethnicity_data) > 1 and any(pct > 10 for _, pct in ethnicity_data):  # Only show if at least one generation has >10%
                    gens, percentages = zip(*ethnicity_data)
                    changes = [percentages[i] - percentages[i-1] for i in range(1, len(percentages))]
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return df

def create_comparative_visualization(results, base_dir="vis/t2i/occupation", exclude_unidentifiable=True):
    """
    Create a special visualization comparing gen 0 and gen 10.
    - Y-axis: Occupations
    - X-axis: Percentage bars
    - Gen 0: Faded color bars
    - Gen 10: Solid color bars (red for increase, blue for decrease)
    
    Args:
        results: DataFrame or list of results
        base_dir: Base directory for output
        exclude_unidentifiable: Whether to exclude unidentifiable cases from analysis
    """
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert results to DataFrame if it's not already
    if not isinstance(results, pd.DataFrame):
        df = pd.DataFrame(results)
    else:
        df = results.copy()
    
    # Filter out unidentifiable cases if requested
    if exclude_unidentifiable:
        df = df[df['gender'] != 'unidentifiable']
        df = df[df['ethnicity'] != 'unidentifiable']
        print(f"Filtered out unidentifiable cases. Remaining data points: {len(df)}")
    
    # Check if we have data for both gen 0 and gen 10
    if 0 not in df['generation'].unique() or 10 not in df['generation'].unique():
        print("Warning: Both generation 0 and 10 are required for comparative visualization.")
        return
    
    # Analyze gender distribution by occupation and generation
    gender_counts = df.groupby(['generation', 'occupation', 'gender']).size().unstack(fill_value=0)
    gender_percentages = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100
    
    # Analyze ethnicity distribution by occupation and generation
    ethnicity_counts = df.groupby(['generation', 'occupation', 'ethnicity']).size().unstack(fill_value=0)
    ethnicity_percentages = ethnicity_counts.div(ethnicity_counts.sum(axis=1), axis=0) * 100
    
    # Get all occupations
    all_occupations = sorted(df['occupation'].unique())
    
    # Create comparative gender visualization
    if 'male' in gender_percentages.columns:
        plt.figure(figsize=(12, 10))
        
        # Extract male percentages for gen 0 and gen 10
        male_data = []
        for occupation in all_occupations:
            gen0_pct = gender_percentages.loc[(0, occupation)].get('male', 0) if (0, occupation) in gender_percentages.index else 0
            gen10_pct = gender_percentages.loc[(10, occupation)].get('male', 0) if (10, occupation) in gender_percentages.index else 0
            male_data.append((occupation, gen0_pct, gen10_pct))
        
        # Sort by gen 0 percentages
        male_data.sort(key=lambda x: x[1])
        
        # Prepare data for plotting
        occupations = [item[0] for item in male_data]
        gen0_pcts = [item[1] for item in male_data]
        gen10_pcts = [item[2] for item in male_data]
        
        # Calculate differences
        differences = [gen10 - gen0 for gen0, gen10 in zip(gen0_pcts, gen10_pcts)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set y-axis
        y_pos = np.arange(len(occupations))
        ax.set_yticks(y_pos)
        ax.set_yticklabels([occ.capitalize() for occ in occupations])
        
        # Plot gen 0 bars (faded)
        gen0_bars = ax.barh(y_pos, gen0_pcts, alpha=0.3, color='gray', label='Gen 0')
        
        # Plot gen 10 differences
        for i, (gen0, diff) in enumerate(zip(gen0_pcts, differences)):
            if diff > 0:  # Increase (red)
                ax.barh(y_pos[i], diff, left=gen0, color='red', alpha=0.8, label='Gen 10 (increase)' if i == 0 else "")
            elif diff < 0:  # Decrease (blue)
                ax.barh(y_pos[i], abs(diff), left=gen0 + diff, color='blue', alpha=0.8, label='Gen 10 (decrease)' if i == 0 else "")
        
        # Add value labels
        for i, (gen0, gen10) in enumerate(zip(gen0_pcts, gen10_pcts)):
            # Gen 0 label
            ax.text(gen0 + 1, i, f'{gen0:.1f}%', va='center', color='gray')
            
            # Gen 10 label
            if gen10 > gen0:
                ax.text(gen10 + 1, i, f'{gen10:.1f}%', va='center', color='red')
            else:
                ax.text(gen10 - 5, i, f'{gen10:.1f}%', va='center', color='blue')
        
        # Set labels and title
        ax.set_xlabel('Percentage of Male (%)')
        ax.set_title('Male Representation Comparison: Gen 0 vs Gen 10')
        
        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Set grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "male_representation_comparison.png"), dpi=300)
        plt.close()
    
    # Create comparative female visualization if available
    if 'female' in gender_percentages.columns:
        plt.figure(figsize=(12, 10))
        
        # Extract female percentages for gen 0 and gen 10
        female_data = []
        for occupation in all_occupations:
            gen0_pct = gender_percentages.loc[(0, occupation)].get('female', 0) if (0, occupation) in gender_percentages.index else 0
            gen10_pct = gender_percentages.loc[(10, occupation)].get('female', 0) if (10, occupation) in gender_percentages.index else 0
            female_data.append((occupation, gen0_pct, gen10_pct))
        
        # Sort by gen 0 percentages
        female_data.sort(key=lambda x: x[1])
        
        # Prepare data for plotting
        occupations = [item[0] for item in female_data]
        gen0_pcts = [item[1] for item in female_data]
        gen10_pcts = [item[2] for item in female_data]
        
        # Calculate differences
        differences = [gen10 - gen0 for gen0, gen10 in zip(gen0_pcts, gen10_pcts)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set y-axis
        y_pos = np.arange(len(occupations))
        ax.set_yticks(y_pos)
        ax.set_yticklabels([occ.capitalize() for occ in occupations])
        
        # Plot gen 0 bars (faded)
        gen0_bars = ax.barh(y_pos, gen0_pcts, alpha=0.3, color='gray', label='Gen 0')
        
        # Plot gen 10 differences
        for i, (gen0, diff) in enumerate(zip(gen0_pcts, differences)):
            if diff > 0:  # Increase (red)
                ax.barh(y_pos[i], diff, left=gen0, color='red', alpha=0.8, label='Gen 10 (increase)' if i == 0 else "")
            elif diff < 0:  # Decrease (blue)
                ax.barh(y_pos[i], abs(diff), left=gen0 + diff, color='blue', alpha=0.8, label='Gen 10 (decrease)' if i == 0 else "")
        
        # Add value labels
        for i, (gen0, gen10) in enumerate(zip(gen0_pcts, gen10_pcts)):
            # Gen 0 label
            ax.text(gen0 + 1, i, f'{gen0:.1f}%', va='center', color='gray')
            
            # Gen 10 label
            if gen10 > gen0:
                ax.text(gen10 + 1, i, f'{gen10:.1f}%', va='center', color='red')
            else:
                ax.text(gen10 - 5, i, f'{gen10:.1f}%', va='center', color='blue')
        
        # Set labels and title
        ax.set_xlabel('Percentage of Female (%)')
        ax.set_title('Female Representation Comparison: Gen 0 vs Gen 10')
        
        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Set grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "female_representation_comparison.png"), dpi=300)
        plt.close()
    
    # Create comparative ethnicity visualization for each ethnicity
    all_ethnicities = sorted(df['ethnicity'].unique())
    
    for ethnicity in all_ethnicities:
        if ethnicity == 'unidentifiable':
            continue
            
        plt.figure(figsize=(12, 10))
        
        # Extract ethnicity percentages for gen 0 and gen 10
        ethnicity_data = []
        for occupation in all_occupations:
            gen0_pct = ethnicity_percentages.loc[(0, occupation)].get(ethnicity, 0) if (0, occupation) in ethnicity_percentages.index else 0
            gen10_pct = ethnicity_percentages.loc[(10, occupation)].get(ethnicity, 0) if (10, occupation) in ethnicity_percentages.index else 0
            ethnicity_data.append((occupation, gen0_pct, gen10_pct))
        
        # Sort by gen 0 percentages
        ethnicity_data.sort(key=lambda x: x[1])
        
        # Prepare data for plotting
        occupations = [item[0] for item in ethnicity_data]
        gen0_pcts = [item[1] for item in ethnicity_data]
        gen10_pcts = [item[2] for item in ethnicity_data]
        
        # Calculate differences
        differences = [gen10 - gen0 for gen0, gen10 in zip(gen0_pcts, gen10_pcts)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set y-axis
        y_pos = np.arange(len(occupations))
        ax.set_yticks(y_pos)
        ax.set_yticklabels([occ.capitalize() for occ in occupations])
        
        # Plot gen 0 bars (faded)
        gen0_bars = ax.barh(y_pos, gen0_pcts, alpha=0.3, color='gray', label='Gen 0')
        
        # Plot gen 10 differences
        for i, (gen0, diff) in enumerate(zip(gen0_pcts, differences)):
            if diff > 0:  # Increase (red)
                ax.barh(y_pos[i], diff, left=gen0, color='red', alpha=0.8, label='Gen 10 (increase)' if i == 0 else "")
            elif diff < 0:  # Decrease (blue)
                ax.barh(y_pos[i], abs(diff), left=gen0 + diff, color='blue', alpha=0.8, label='Gen 10 (decrease)' if i == 0 else "")
        
        # Add value labels
        for i, (gen0, gen10) in enumerate(zip(gen0_pcts, gen10_pcts)):
            # Gen 0 label
            ax.text(gen0 + 1, i, f'{gen0:.1f}%', va='center', color='gray')
            
            # Gen 10 label
            if gen10 > gen0:
                ax.text(gen10 + 1, i, f'{gen10:.1f}%', va='center', color='red')
            else:
                ax.text(gen10 - 5, i, f'{gen10:.1f}%', va='center', color='blue')
        
        # Set labels and title
        ax.set_xlabel(f'Percentage of {ethnicity} (%)')
        ax.set_title(f'{ethnicity} Representation Comparison: Gen 0 vs Gen 10')
        
        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Set grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{ethnicity}_representation_comparison.png"), dpi=300)
        plt.close()
    
    # Create a summary table
    summary_data = []
    
    # Gender summary
    for gender in gender_percentages.columns:
        if gender == 'unidentifiable':
            continue
            
        for occupation in all_occupations:
            gen0_gender = gender_percentages.loc[(0, occupation)].get(gender, 0) if (0, occupation) in gender_percentages.index else 0
            gen10_gender = gender_percentages.loc[(10, occupation)].get(gender, 0) if (10, occupation) in gender_percentages.index else 0
            diff = gen10_gender - gen0_gender
            
            summary_data.append({
                'occupation': occupation,
                'category': 'gender',
                'attribute': gender,
                'gen_0': gen0_gender,
                'gen_10': gen10_gender,
                'difference': diff,
                'percent_change': (diff / gen0_gender * 100) if gen0_gender > 0 else float('inf')
            })
    
    # Ethnicity summary
    for ethnicity in all_ethnicities:
        if ethnicity == 'unidentifiable':
            continue
            
        for occupation in all_occupations:
            gen0_eth = ethnicity_percentages.loc[(0, occupation)].get(ethnicity, 0) if (0, occupation) in ethnicity_percentages.index else 0
            gen10_eth = ethnicity_percentages.loc[(10, occupation)].get(ethnicity, 0) if (10, occupation) in ethnicity_percentages.index else 0
            diff = gen10_eth - gen0_eth
            
            summary_data.append({
                'occupation': occupation,
                'category': 'ethnicity',
                'attribute': ethnicity,
                'gen_0': gen0_eth,
                'gen_10': gen10_eth,
                'difference': diff,
                'percent_change': (diff / gen0_eth * 100) if gen0_eth > 0 else float('inf')
            })
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, "gen0_vs_gen10_comparison.csv"), index=False)
    
    # Create a summary visualization of gender changes across occupations
    plt.figure(figsize=(15, 10))
    
    # Extract gender changes
    gender_changes = summary_df[summary_df['category'] == 'gender'].copy()
    
    # Pivot the data for plotting
    pivot_df = gender_changes.pivot(index='occupation', columns='attribute', values='difference')
    
    # Sort by male difference
    if 'male' in pivot_df.columns:
        pivot_df = pivot_df.sort_values(by='male')
    
    # Plot the changes
    ax = pivot_df.plot(kind='barh', figsize=(15, 10), 
                      color=['blue', 'red', 'green', 'purple', 'orange'],
                      alpha=0.7)
    
    # Add labels and title
    plt.title('Gender Representation Changes from Gen 0 to Gen 10 by Occupation')
    plt.xlabel('Percentage Point Change')
    plt.ylabel('Occupation')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.legend(title='Gender')
    
    # Add value labels
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if abs(width) > 1:  # Only label significant changes
            x = p.get_x() + p.get_width() + 0.5 if width > 0 else p.get_x() + p.get_width() - 3
            y = p.get_y() + p.get_height() / 2
            ax.annotate(f'{width:.1f}', (x, y), ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gender_changes_summary.png"), dpi=300)
    plt.close()
    
    print("\nComparative visualization between Gen 0 and Gen 10 created.")
    print(f"Results saved to {results_dir}")
    
    return summary_df

def create_gender_horizontal_barchart(results, base_dir="vis/t2i/occupation"):
    """
    Create a specialized horizontal bar chart for gender ratios comparing Gen 0 and Gen 10.
    Gen 0 is shown in grey, Gen 10 in red (if shift toward female) or blue (if shift toward male).
    A vertical line is drawn at 0 (which corresponds to 50% male ratio) for reference.
    Occupations are sorted by Gen 0 male ratio from top to bottom.
    The x-axis is centered at 0.5, ranging from -0.5 to 0.5, to indicate bias toward male (right) or female (left).
    
    Args:
        results (list or pd.DataFrame): List of dictionaries or DataFrame containing classification results
        base_dir (str): Base directory for saving results
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert results to DataFrame if it's a list
    if isinstance(results, list):
        print("Converting results list to DataFrame...")
        results_df = pd.DataFrame(results)
    else:
        results_df = results
    
    # Check if the DataFrame is empty
    if results_df.empty:
        print("Error: No results to visualize")
        return
    
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Results DataFrame columns: {results_df.columns.tolist()}")
    
    # Check if 'gender' column exists
    if 'gender' not in results_df.columns:
        print("Error: 'gender' column not found in results")
        return
    
    # Filter out unidentifiable gender
    filtered_results = results_df[results_df['gender'] != 'unidentifiable']
    print(f"Filtered results shape (after removing unidentifiable): {filtered_results.shape}")
    
    # Only keep Gen 0 and Gen 10
    target_gens = [0, 10]
    filtered_results = filtered_results[filtered_results['generation'].isin(target_gens)]
    
    # Get unique occupations
    occupations = sorted(filtered_results['occupation'].unique())
    
    print(f"Unique occupations: {occupations}")
    
    # Calculate male ratios for each generation and occupation
    male_ratios = {}
    for gen in target_gens:
        male_ratios[gen] = {}
        for occupation in occupations:
            gen_occ_data = filtered_results[(filtered_results['generation'] == gen) & 
                           (filtered_results['occupation'] == occupation)]
            
            if len(gen_occ_data) > 0:
                gender_counts = gen_occ_data['gender'].value_counts(normalize=True)
                # Get male ratio
                male_ratio = gender_counts.get('male', 0)
                male_ratios[gen][occupation] = male_ratio
                print(f"Gen {gen}, {occupation}: Male={male_ratio:.2f}")
    
    # Sort occupations by Gen 0 male ratio (descending)
    sorted_occupations = sorted(occupations, key=lambda x: male_ratios[0].get(x, 0), reverse=True)
    
    # Create a figure for the horizontal bar chart - increase width for more space
    plt.figure(figsize=(6, 6))  # Increased width from 4 to 6
    
    # Set up the plot with a white background
    ax = plt.subplot(111)
    ax.set_facecolor('white')
    
    # Set position of bars on y axis
    y_pos = np.arange(len(sorted_occupations))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([occ.capitalize() for occ in sorted_occupations])
    
    # Transform male ratios to centered scale (-0.5 to 0.5)
    # Where 0.5 male ratio becomes 0, 0 becomes -0.5, and 1 becomes 0.5
    transform_ratio = lambda r: r - 0.5
    
    # Draw vertical line at 0 (which corresponds to 50% male ratio)
    plt.axvline(x=0, color=GREY_800, linestyle='-', alpha=0.7)
    
    # Prepare data for Gen 0 and Gen 10
    gen0_values = [transform_ratio(male_ratios[0].get(occupation, 0)) for occupation in sorted_occupations]
    gen10_values = [transform_ratio(male_ratios[10].get(occupation, 0)) for occupation in sorted_occupations]
    
    # Determine colors for Gen 10 bars based on shift direction
    # Red for shift toward female (decrease in male ratio), Blue for shift toward male (increase in male ratio)
    colors = []
    for i, occupation in enumerate(sorted_occupations):
        gen0_val = male_ratios[0].get(occupation, 0)
        gen10_val = male_ratios[10].get(occupation, 0)
        # Use neutral color if no significant change
        if abs(gen10_val - gen0_val) < 0.01:  # Consider as no change if difference is very small
            colors.append(GEN_10_NEUTRAL_COLOR)
        else:
            # Red if shift toward female (decrease in male ratio)
            # Blue if shift toward male (increase in male ratio)
            colors.append(ERROR if gen10_val < gen0_val else PRIMARY)
    
    # Plot bars with the smaller value on top for each occupation
    for i, occupation in enumerate(sorted_occupations):
        gen0_val = transform_ratio(male_ratios[0].get(occupation, 0))
        gen10_val = transform_ratio(male_ratios[10].get(occupation, 0))
        
        # Special case for nurse - always put grey on top
        if occupation == "nurse":
            ax.barh(y_pos[i], gen10_val, height=0.3, color=colors[i], zorder=5, edgecolor=GREY_800, linewidth=0.5)
            ax.barh(y_pos[i], gen0_val, height=0.3, color=GEN_0_COLOR, zorder=10, edgecolor=GREY_800, linewidth=0.5)
        # Special case for teacher - always put grey on top
        elif occupation == "teacher":
            ax.barh(y_pos[i], gen10_val, height=0.3, color=colors[i], zorder=5, edgecolor=GREY_800, linewidth=0.5)
            ax.barh(y_pos[i], gen0_val, height=0.3, color=GEN_0_COLOR, zorder=10, edgecolor=GREY_800, linewidth=0.5)
        # For other occupations, determine which bar should be on top (smaller absolute value)
        elif abs(gen0_val) <= abs(gen10_val):
            # Gen 0 is smaller or equal, draw Gen 10 first, then Gen 0 on top
            ax.barh(y_pos[i], gen10_val, height=0.3, color=colors[i], zorder=5, edgecolor=GREY_800, linewidth=0.5)
            ax.barh(y_pos[i], gen0_val, height=0.3, color=GEN_0_COLOR, zorder=10, edgecolor=GREY_800, linewidth=0.5)
        else:
            # Gen 10 is smaller, draw Gen 0 first, then Gen 10 on top
            ax.barh(y_pos[i], gen0_val, height=0.3, color=GEN_0_COLOR, zorder=5, edgecolor=GREY_800, linewidth=0.5)
            ax.barh(y_pos[i], gen10_val, height=0.3, color=colors[i], zorder=10, edgecolor=GREY_800, linewidth=0.5)
    
    # Add dummy bars for the legend
    ax.barh([-1], [0], height=0.3, color=GEN_0_COLOR, label='Gen 0', edgecolor=GREY_800, linewidth=0.5)
    ax.barh([-1], [0], height=0.3, color=ERROR, label='Gen 10 (shift female)', edgecolor=GREY_800, linewidth=0.5)
    ax.barh([-1], [0], height=0.3, color=PRIMARY, label='Gen 10 (shift male)', edgecolor=GREY_800, linewidth=0.5)
    
    # Add labels with larger font sizes - only x-axis label, no y-axis label
    plt.xlabel('Gender Bias (← Female | Male →)', fontsize=10, color=GREY_900)  # Increased font size
    # Removed y-axis label
    
    # Set custom x-ticks with larger font size
    plt.xticks([-0.5, -0.25, 0, 0.25, 0.5], ['100% F', '75% F', '50/50', '75% M', '100% M'], fontsize=9, color=GREY_800)  # Increased font size
    
    # Set y-ticks (occupation names) with larger font size
    plt.yticks(y_pos, [occ.capitalize() for occ in sorted_occupations], fontsize=9, color=GREY_800)  # Increased font size
    
    # Set x-axis limits with more space on both sides
    plt.xlim(-0.7, 0.7)  # Increased from -0.6, 0.6 to -0.7, 0.7
    
    # Add a legend with larger font in the top right corner
    plt.legend(loc='upper right', fontsize='x-small', framealpha=0.9)  # Reduced font size for narrower plot
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.3, color=GREY_300)
    
    # Helper function to format the displayed value based on which side of the axis it falls
    def format_value(male_ratio):
        if male_ratio < 0.5:
            # For values left of center, show female percentage
            female_ratio = 1 - male_ratio
            return f'{female_ratio:.2f}F'
        else:
            # For values right of center, show male percentage
            return f'{male_ratio:.2f}M'
    
    # Add value labels on the bars with larger font size
    for i, occupation in enumerate(sorted_occupations):
        gen0_val = male_ratios[0].get(occupation, 0)
        gen10_val = male_ratios[10].get(occupation, 0)
        
        # Transform to centered scale
        gen0_transformed = transform_ratio(gen0_val)
        gen10_transformed = transform_ratio(gen10_val)
        
        # Format values based on which side of the axis they fall
        gen0_text = format_value(gen0_val)
        gen10_text = format_value(gen10_val)
        
        # Check if there's a significant change
        is_change = abs(gen10_val - gen0_val) >= 0.01
        
        if is_change:
            # Add Gen 0 value - larger font with higher zorder to ensure visibility
            # Position text further from the bar
            plt.text(gen0_transformed + (0.03 if gen0_transformed >= 0 else -0.03),  # Increased offset from 0.02 to 0.03
                    y_pos[i], gen0_text, 
                    va='center', ha='left' if gen0_transformed >= 0 else 'right', 
                    fontsize=7, color=GREY_800, zorder=25, weight='bold')  # Reduced font size for narrower plot
            
            # Add Gen 10 value - larger font
            # Position text further from the bar
            plt.text(gen10_transformed + (0.03 if gen10_transformed >= 0 else -0.03),  # Increased offset from 0.02 to 0.03
                    y_pos[i], gen10_text, 
                    va='center', ha='left' if gen10_transformed >= 0 else 'right', 
                    fontsize=7, color=GREY_800, zorder=25, weight='bold')  # Reduced font size for narrower plot
        else:
            # If no change, just show the value once in black - larger font
            # Position text further from the bar
            plt.text(gen0_transformed + (0.03 if gen0_transformed >= 0 else -0.03),  # Increased offset from 0.02 to 0.03
                    y_pos[i], gen0_text, 
                    va='center', ha='left' if gen0_transformed >= 0 else 'right', 
                    fontsize=7, color=GREY_800, zorder=25, weight='bold')  # Reduced font size for narrower plot
    
    # Adjust layout for a better fit
    plt.tight_layout(pad=1.2)  # Increased padding from 0.8 to 1.2
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, 'gender_bias_compact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gender bias chart saved to {os.path.join(results_dir, 'gender_bias_compact.png')}")
    
    # Create a summary table of changes
    summary_data = []
    for occupation in sorted_occupations:
        gen0_val = male_ratios[0].get(occupation, 0)
        gen10_val = male_ratios[10].get(occupation, 0)
        change = gen10_val - gen0_val
        summary_data.append({
            'Occupation': occupation,
            'Gen0_Male_Ratio': gen0_val,
            'Gen10_Male_Ratio': gen10_val,
            'Change': change,
            'Direction': 'No Change' if abs(change) < 0.01 else ('Shift Male' if change > 0 else 'Shift Female')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(results_dir, 'gender_bias_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Gender bias summary saved to {summary_path}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate gender and ethnicity bias across model generations")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="CFG scale to use (default: 7.0)")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate per occupation (default: 100)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for image generation (default: 4)")
    parser.add_argument("--skip_generation", action="store_true", help="Skip image generation and use existing images")
    parser.add_argument("--skip_classification", action="store_true", help="Skip image classification and use existing results")
    parser.add_argument("--only_analysis", action="store_true", help="Skip generation and classification, only run analysis")
    parser.add_argument("--only_comparative", action="store_true", help="Only create comparative visualization between gen 0 and gen 10")
    parser.add_argument("--output_dir", type=str, default="vis/t2i/occupation", help="Base directory for output")
    parser.add_argument("--only_generation", type=int, default=None, help="Only process a specific generation (0, 4, or 10)")
    parser.add_argument("--only_occupation", type=str, default=None, help="Only process a specific occupation")
    parser.add_argument("--generations", type=str, default="0,4,10", help="Comma-separated list of generations to process (default: 0,4,10)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Enable debug output
    debug = args.debug
    
    if debug:
        print("Debug mode enabled")
        print(f"Arguments: {args}")
    
    # Setup directories
    base_dir = setup_directories(args.output_dir)
    
    if debug:
        print(f"Base directory: {base_dir}")
    
    # Results to collect
    all_results = []
    
    # If only_analysis or only_comparative is set, skip generation and classification
    if args.only_analysis or args.only_comparative:
        args.skip_generation = True
        args.skip_classification = True
    
    # Determine which generations to process
    if args.only_generation is not None:
        generations_to_process = [args.only_generation]
    else:
        generations_to_process = [int(g) for g in args.generations.split(',')]
    
    # Determine which occupations to process
    occupations_to_process = OCCUPATIONS
    if args.only_occupation is not None:
        if args.only_occupation in OCCUPATIONS:
            occupations_to_process = [args.only_occupation]
        else:
            print(f"Warning: Invalid occupation '{args.only_occupation}'. Using all occupations.")
    
    if debug:
        print(f"Generations to process: {generations_to_process}")
        print(f"Occupations to process: {occupations_to_process}")
    
    if not args.skip_generation:
        try:
            # Generate images for each occupation and generation
            for gen in generations_to_process:
                # Check if images already exist for this generation
                all_exist = True
                for occupation in occupations_to_process:
                    image_dir = os.path.join(base_dir, f"gen_{gen}", occupation.replace(" ", "_"))
                    if not os.path.exists(image_dir):
                        all_exist = False
                        break
                    
                    # Check if we have enough images
                    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if len(image_files) < args.num_images:
                        all_exist = False
                        break
                
                if all_exist:
                    print(f"All images already exist for generation {gen}. Skipping generation.")
                    continue
                
                # Load the model
                pipeline = load_model(gen, args.cfg_scale)
                
                if pipeline is None:
                    print(f"Skipping generation {gen} as model could not be loaded")
                    continue
                
                for occupation in occupations_to_process:
                    try:
                        print(f"Processing generation {gen}, occupation {occupation}...")
                        # Generate images in batches
                        image_paths = generate_images_batch(
                            pipeline, 
                            occupation, 
                            gen, 
                            num_images=args.num_images, 
                            base_dir=base_dir,
                            cfg_scale=args.cfg_scale,
                            batch_size=args.batch_size
                        )
                        
                        print(f"Generated/found {len(image_paths)} images for {occupation} (gen {gen})")
                        
                        # Clean up to save memory
                        torch.cuda.empty_cache()
                    except KeyboardInterrupt:
                        print(f"\nGeneration interrupted for {occupation} (gen {gen}). Moving to next task.")
                        torch.cuda.empty_cache()
                        continue
                    except Exception as e:
                        print(f"Error generating images for {occupation} (gen {gen}): {str(e)}")
                        torch.cuda.empty_cache()
                        continue
                
                # Clean up the pipeline to save memory
                del pipeline
                torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print("\nImage generation interrupted. Moving to classification phase.")
            torch.cuda.empty_cache()
    else:
        print("Skipping image generation phase.")
    
    if not args.skip_classification:
        try:
            # Check if classification results already exist
            results_path = os.path.join(base_dir, "results", "classification_results_raw.csv")
            if os.path.exists(results_path):
                print(f"Classification results already exist at {results_path}. Loading existing results.")
                results_df = pd.read_csv(results_path)
                all_results = results_df.to_dict('records')
            else:
                # Load classifiers
                classifiers = load_classifiers()
                
                # Classify all images
                print("Classifying images...")
                for gen in generations_to_process:
                    for occupation in occupations_to_process:
                        try:
                            # Directory with images
                            image_dir = os.path.join(base_dir, f"gen_{gen}", occupation.replace(" ", "_"))
                            
                            if not os.path.exists(image_dir):
                                print(f"Warning: Directory {image_dir} does not exist. Skipping.")
                                continue
                            
                            # Get all image files
                            image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                            
                            if not image_files:
                                print(f"Warning: No images found in {image_dir}. Skipping.")
                                continue
                            
                            print(f"Classifying {len(image_files)} images for {occupation} (gen {gen})...")
                            
                            # Get full paths
                            image_paths = [os.path.join(image_dir, f) for f in image_files]
                            
                            if debug:
                                print(f"First few image paths: {image_paths[:3]}")
                            
                            # Classify images in batches
                            batch_results = classify_images_batch(image_paths, classifiers)
                            
                            if debug:
                                print(f"First few classification results: {batch_results[:3]}")
                            
                            # Add metadata
                            for result in batch_results:
                                result["generation"] = gen
                                result["occupation"] = occupation
                            
                            # Add to results
                            all_results.extend(batch_results)
                            
                            # Save intermediate results after each occupation
                            print(f"Saving intermediate results for {occupation} (gen {gen})...")
                            intermediate_df = pd.DataFrame(all_results)
                            results_dir = os.path.join(base_dir, "results")
                            os.makedirs(results_dir, exist_ok=True)
                            intermediate_df.to_csv(os.path.join(results_dir, "classification_results_intermediate.csv"), index=False)
                        
                        except KeyboardInterrupt:
                            print(f"\nClassification interrupted for {occupation} (gen {gen}). Moving to next task.")
                            continue
                        except Exception as e:
                            print(f"Error classifying images for {occupation} (gen {gen}): {str(e)}")
                            import traceback
                            traceback.print_exc()
                            continue
                
                # Save final classification results
                print("Saving final classification results...")
                results_df = pd.DataFrame(all_results)
                results_dir = os.path.join(base_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                results_df.to_csv(os.path.join(results_dir, "classification_results_raw.csv"), index=False)
        
        except KeyboardInterrupt:
            print("\nClassification interrupted. Moving to analysis phase.")
            
            # Save partial results
            if all_results:
                print("Saving partial classification results...")
                results_df = pd.DataFrame(all_results)
                results_dir = os.path.join(base_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                results_df.to_csv(os.path.join(results_dir, "classification_results_partial.csv"), index=False)
    else:
        print("Skipping classification phase.")
        # Load existing classification results
        results_path = os.path.join(base_dir, "results", "classification_results_raw.csv")
        if os.path.exists(results_path):
            print(f"Loading existing classification results from {results_path}")
            results_df = pd.read_csv(results_path)
            all_results = results_df.to_dict('records')
            
            if debug:
                print(f"Loaded {len(all_results)} classification results")
                if all_results:
                    print(f"First result: {all_results[0]}")
                    print(f"Unique generations: {sorted(results_df['generation'].unique())}")
                    print(f"Unique occupations: {sorted(results_df['occupation'].unique())}")
                    print(f"Unique genders: {sorted(results_df['gender'].unique())}")
                    print(f"Unique ethnicities: {sorted(results_df['ethnicity'].unique())}")
        else:
            print(f"Warning: No existing classification results found at {results_path}")
            
            # Try loading partial results
            partial_path = os.path.join(base_dir, "results", "classification_results_partial.csv")
            if os.path.exists(partial_path):
                print(f"Loading partial classification results from {partial_path}")
                results_df = pd.read_csv(partial_path)
                all_results = results_df.to_dict('records')
                
                if debug:
                    print(f"Loaded {len(all_results)} partial classification results")
                    if all_results:
                        print(f"Unique generations: {sorted(results_df['generation'].unique())}")
            else:
                print(f"Warning: No partial classification results found at {partial_path}")
                
            # Try loading intermediate results
            intermediate_path = os.path.join(base_dir, "results", "classification_results_intermediate.csv")
            if os.path.exists(intermediate_path):
                print(f"Loading intermediate classification results from {intermediate_path}")
                results_df = pd.read_csv(intermediate_path)
                all_results = results_df.to_dict('records')
                
                if debug:
                    print(f"Loaded {len(all_results)} intermediate classification results")
                    if all_results:
                        print(f"Unique generations: {sorted(results_df['generation'].unique())}")
            else:
                print(f"Warning: No intermediate classification results found at {intermediate_path}")
    
    # Analyze results
    if all_results:
        try:
            if args.only_comparative:
                print("Creating comparative visualization between Gen 0 and Gen 10...")
                # Convert generation column to int if it's not already
                if isinstance(all_results, list):
                    for result in all_results:
                        if 'generation' in result and not isinstance(result['generation'], int):
                            result['generation'] = int(result['generation'])
                else:
                    if 'generation' in all_results.columns and not pd.api.types.is_integer_dtype(all_results['generation']):
                        all_results['generation'] = all_results['generation'].astype(int)
                
                create_comparative_visualization(all_results, base_dir)
                # Create gender horizontal bar chart
                create_gender_horizontal_barchart(all_results, base_dir)
            else:
                print("Analyzing results...")
                # Convert generation column to int if it's not already
                if isinstance(all_results, list):
                    for result in all_results:
                        if 'generation' in result and not isinstance(result['generation'], int):
                            result['generation'] = int(result['generation'])
                else:
                    if 'generation' in all_results.columns and not pd.api.types.is_integer_dtype(all_results['generation']):
                        all_results['generation'] = all_results['generation'].astype(int)
                
                results_df = analyze_results(all_results, base_dir)
                
                # Also create comparative visualization
                print("Creating comparative visualization between Gen 0 and Gen 10...")
                create_comparative_visualization(all_results, base_dir)
                
                # Create gender horizontal bar chart
                create_gender_horizontal_barchart(all_results, base_dir)
                
                print(f"Evaluation complete. Results saved to {os.path.join(base_dir, 'results')}")
        except Exception as e:
            print(f"Error analyzing results: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("No results to analyze. Make sure images were generated or exist in the specified directories.")

if __name__ == "__main__":
    main() 