"""
Generate a manual labeling template CSV for the 23 scenes.
"""
import pandas as pd

SCENE_NAMES = [
    "abandoned", "abandoned-demo", "abandoned-flipped", "cubetest", 
    "fantasticvillage-open", "lightfoliage", "lightfoliage-close", 
    "oldmine", "oldmine-close", "oldmine-warm", "quarry-all", 
    "quarry-rocksonly", "resto-close", "resto-fwd", "resto-pan", 
    "scifi", "subway-lookdown", "subway-turn", "wildwest-bar", 
    "wildwest-barzoom", "wildwest-behindcounter", "wildwest-store", 
    "wildwest-town"
]

def create_labeling_template():
    """Create a CSV template for manual scene labeling."""
    
    template_data = []
    
    for scene in SCENE_NAMES:
        template_data.append({
            'scene': scene,
            'environment': '',  # e.g., abandoned_building, forest, urban, quarry, western
            'motion_type': '',  # Options: static, slow_pan, moderate_motion, fast_motion, complex_motion
            'motion_speed': '',  # Options: very_slow, slow, moderate, fast, very_fast
            'content_type': '',  # Options: geometric, organic, mixed, architectural
            'has_vegetation': '',  # Options: none, sparse, moderate, dense
            'has_particles': '',  # Options: yes, no
            'visual_complexity': '',  # Options: simple, moderate, complex, very_complex
            'texture_detail': '',  # Options: low, moderate, high, very_high
            'lighting': '',  # Options: bright, moderate, dark, mixed
            'contrast': '',  # Options: low, moderate, high
            'has_reflections': '',  # Options: yes, no
            'has_transparency': '',  # Options: yes, no
            'notes': ''  # Any additional observations
        })
    
    df = pd.DataFrame(template_data)
    return df

if __name__ == "__main__":
    df = create_labeling_template()
    
    output_file = "manual_labels_template.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Created manual labeling template: {output_file}")
    print(f"\nTotal scenes to label: {len(df)}")
    print("\nInstructions:")
    print("="*80)
    print("1. Open manual_labels_template.csv in Excel or a text editor")
    print("2. Fill in each column for all 23 scenes by watching the 16SSAA.mp4 videos")
    print("3. Use the suggested options in the comments, or add your own values")
    print("4. Save the completed file as 'manual_labels_completed.csv'")
    print("\nColumn Guidelines:")
    print("-" * 80)
    print("environment: General environment type (e.g., 'abandoned_building', 'quarry', 'western_town')")
    print("motion_type: Overall camera motion pattern")
    print("motion_speed: Subjective speed rating")
    print("content_type: Primary type of visual content")
    print("has_vegetation: Amount of foliage/plants in scene")
    print("has_particles: Presence of particle effects (dust, smoke, etc.)")
    print("visual_complexity: Overall visual busyness")
    print("texture_detail: Amount of fine surface detail")
    print("lighting: Overall brightness level")
    print("contrast: Scene contrast level")
    print("has_reflections: Reflective surfaces present")
    print("has_transparency: Glass or transparent materials present")
    print("notes: Any other relevant observations")
    
    print("\n" + "="*80)
    print("Preview of template:")
    print(df.head(5).to_string(index=False))