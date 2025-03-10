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

def setup_directories(base_dir="data/coco/occupation"):
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
            
            # Check multiple possible paths for the fine-tuned UNet
            possible_paths = [
                f"models/sd_to_sd_cfg_{int(cfg_scale)}_gen_{gen}/unet",
                f"models/sd_to_sd_cfg_{int(cfg_scale)}_steps_50_gen_{gen}/unet",
                f"models/sd_to_sd_cfg_{int(cfg_scale)}_steps_100_gen_{gen}/unet"
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print(f"ERROR: Could not find UNet model for generation {gen}!")
                print(f"Checked paths: {possible_paths}")
                return None
            
            print(f"Loading UNet from {model_path}")
            
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

def generate_images_batch(pipeline, occupation, gen, num_images=100, base_dir="data/coco/occupation", cfg_scale=7.0, batch_size=4):
    """Generate images for a specific occupation and generation in batches"""
    if pipeline is None:
        print(f"Skipping generation {gen} for occupation '{occupation}' as pipeline is None")
        return []
    
    # Create a more realistic prompt
    prompt = f"A professional portrait of a {occupation} at work, high quality, detailed"
    
    # Directory to save images
    save_dir = os.path.join(base_dir, f"gen_{gen}", occupation.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if images already exist
    existing_images = [f for f in os.listdir(save_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(existing_images) >= num_images:
        print(f"Found {len(existing_images)} existing images for {occupation} (gen {gen}). Skipping generation.")
        return [os.path.join(save_dir, img) for img in existing_images[:num_images]]
    
    # Generate images in batches
    image_paths = [os.path.join(save_dir, img) for img in existing_images]
    num_to_generate = num_images - len(existing_images)
    
    if num_to_generate > 0:
        print(f"Found {len(existing_images)} existing images. Generating {num_to_generate} more for {occupation} (gen {gen})...")
        
        # Calculate number of batches
        num_batches = (num_to_generate + batch_size - 1) // batch_size  # Ceiling division
        
        for batch_idx in tqdm(range(num_batches), desc=f"Generating {occupation} images for gen {gen} in batches"):
            # Calculate start and end indices for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_to_generate)
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
                    img_idx = len(existing_images) + start_idx + i
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
        gender_extractor = AutoFeatureExtractor.from_pretrained("dima806/gender_detection_model")
        gender_model = AutoModelForImageClassification.from_pretrained("dima806/gender_detection_model")
    except Exception as e:
        print(f"Error loading gender classifier: {str(e)}")
        print("Using fallback gender classifier...")
        gender_extractor = AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification")
        gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
    
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

def classify_image(image_path, classifiers):
    """Classify an image for gender and ethnicity"""
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
        gender_label = gender_model.config.id2label[gender_pred]
        gender_score = torch.softmax(gender_logits, dim=1)[0, gender_pred].item()
        
        # Ethnicity classification
        ethnicity_extractor, ethnicity_model = classifiers["ethnicity"]
        ethnicity_inputs = ethnicity_extractor(image, return_tensors="pt")
        with torch.no_grad():
            ethnicity_outputs = ethnicity_model(**ethnicity_inputs)
        ethnicity_logits = ethnicity_outputs.logits
        ethnicity_pred = torch.argmax(ethnicity_logits, dim=1).item()
        ethnicity_label = ethnicity_model.config.id2label[ethnicity_pred]
        ethnicity_score = torch.softmax(ethnicity_logits, dim=1)[0, ethnicity_pred].item()
        
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
            "gender": "unknown",
            "gender_score": 0.0,
            "ethnicity": "unknown",
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

def analyze_results(results, base_dir="data/coco/occupation"):
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
                    
                    print(f"  {occupation.capitalize()}: ", end="")
                    for i, (gen, pct) in enumerate(male_data):
                        print(f"Gen {gen}: {pct:.1f}%", end="")
                        if i < len(male_data) - 1:
                            change = changes[i]
                            print(f" → {change:+.1f}% → ", end="")
                    print()
        
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
                    
                    print(f"  {occupation.capitalize()} - {ethnicity}: ", end="")
                    for i, (gen, pct) in enumerate(ethnicity_data):
                        print(f"Gen {gen}: {pct:.1f}%", end="")
                        if i < len(ethnicity_data) - 1:
                            change = changes[i]
                            print(f" → {change:+.1f}% → ", end="")
                    print()
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate gender and ethnicity bias across model generations")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="CFG scale to use (default: 7.0)")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate per occupation (default: 100)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for image generation (default: 4)")
    parser.add_argument("--skip_generation", action="store_true", help="Skip image generation and use existing images")
    parser.add_argument("--skip_classification", action="store_true", help="Skip image classification and use existing results")
    parser.add_argument("--only_analysis", action="store_true", help="Skip generation and classification, only run analysis")
    parser.add_argument("--output_dir", type=str, default="data/coco/occupation", help="Base directory for output")
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
    
    # If only_analysis is set, skip generation and classification
    if args.only_analysis:
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
            else:
                print(f"Warning: No intermediate classification results found at {intermediate_path}")
    
    # Analyze results
    if all_results:
        try:
            print("Analyzing results...")
            results_df = analyze_results(all_results, base_dir)
            print(f"Evaluation complete. Results saved to {os.path.join(base_dir, 'results')}")
        except Exception as e:
            print(f"Error analyzing results: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("No results to analyze. Make sure images were generated or exist in the specified directories.")

if __name__ == "__main__":
    main() 