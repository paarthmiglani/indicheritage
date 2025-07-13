import os
import json

def create_metadata_file(image_dir="ocr/test_images", output_file="ocr/metadata.jsonl"):
    """
    Scans a directory for image and text files, matches them, and creates a metadata file.

    The script assumes the following naming convention:
    - Image files: img_XXX.jpg
    - Annotation files: gt_img_XXX.txt

    Args:
        image_dir (str): The directory containing the images and annotation files.
        output_file (str): The path to the output metadata file (in JSON Lines format).
    """
    print(f"Scanning directory: {image_dir}")

    image_files = [f for f in os.listdir(image_dir) if f.startswith("img_") and f.endswith(".jpg")]

    if not image_files:
        print(f"WARNING: No image files with the pattern 'img_*.jpg' found in {image_dir}.")
        print("Please ensure your dataset is in the correct directory and follows the naming convention.")
        return

    metadata = []
    not_found_count = 0

    for img_filename in image_files:
        # Construct the corresponding annotation filename
        base_name = img_filename.replace("img_", "").replace(".jpg", "")
        gt_filename = f"gt_img_{base_name}.txt"
        gt_filepath = os.path.join(image_dir, gt_filename)

        img_filepath = os.path.join(image_dir, img_filename)

        if os.path.exists(gt_filepath):
            try:
                with open(gt_filepath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                if text: # Ensure the annotation is not empty
                    metadata.append({
                        "file_name": img_filepath,
                        "text": text
                    })
                else:
                    print(f"Skipping empty annotation file: {gt_filename}")

            except Exception as e:
                print(f"Error reading file {gt_filepath}: {e}")
        else:
            print(f"Annotation file not found for image {img_filename}")
            not_found_count += 1

    if not metadata:
        print("No matching image-annotation pairs were found. Cannot create metadata file.")
        return

    print(f"\nFound {len(metadata)} matching image-annotation pairs.")
    if not_found_count > 0:
        print(f"Could not find annotations for {not_found_count} images.")

    # Write the metadata to a .jsonl file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in metadata:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\nSuccessfully created metadata file at: {output_file}")
        print(f"Total lines written: {len(metadata)}")
    except Exception as e:
        print(f"Error writing metadata file: {e}")

if __name__ == "__main__":
    # Assuming the script is run from the root of the repository
    create_metadata_file()
