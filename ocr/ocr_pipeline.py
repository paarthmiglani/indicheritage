import torch
from PIL import Image, ImageDraw
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoFeatureExtractor
import os

# --- Model Analysis and Documentation ---
# Model Tested: QuickHawk/trocr-indic
#
# Observations:
# 1. The model loads and runs correctly using the transformers library.
# 2. When tested on Devanagari Sanskrit text (isha_upanishad_sample.jpg) and a custom image (img_1899.jpg),
#    the model did not produce accurate Devanagari output.
# 3. The output characters were predominantly from the Eastern Nagari script family (Bengali/Assamese).
#    For example, for an image containing Devanagari text, it output "ৱাৰ্ডৰ" (Assamese/Bengali script).
#
# Conclusion:
# The model card for QuickHawk/trocr-indic is misleading. It claims training on many Indic languages
# but also mentions a limitation of being trained with "only Devanagari Scripts". Our testing shows
# the output is heavily biased towards a script other than Devanagari.
#
# Recommendation:
# This model is NOT suitable for general-purpose Devanagari or multi-script Indic OCR out-of-the-box.
# It would require significant fine-tuning on a correctly prepared Devanagari (or other target script) dataset.
# The pipeline code itself is a valid baseline for running a TrOCR model, but the model choice needs to be reconsidered.
# For the purpose of this project, we will leave this pipeline as a baseline demonstration.
# Future work should focus on fine-tuning or finding a more reliable pre-trained model.
# ---

class IndicOCR:
    def __init__(self, model_name="QuickHawk/trocr-indic"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None

        try:
            print(f"Loading model: {self.model_name}...")
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            print(f"Loading processor for: {self.model_name}...")

            try:
                self.processor = TrOCRProcessor.from_pretrained(self.model_name)
                print("Successfully loaded TrOCRProcessor.")
            except Exception as e_trocr_proc:
                print(f"Failed to load TrOCRProcessor directly: {e_trocr_proc}.")
                print("Attempting to load tokenizer and feature extractor separately...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                print("Successfully loaded tokenizer and feature extractor separately.")

            print("Model and associated components loaded successfully.")
        except Exception as e:
            print(f"Error loading model or components: {e}")
            self.model = None
            self.processor = None
            self.tokenizer = None
            self.feature_extractor = None

    def _process_image_manual(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a path string or a PIL Image object")

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        return pixel_values.to(self.device)

    def _decode_text_manual(self, generated_ids):
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if generated_text.startswith("<") and ">" in generated_text:
            parts = generated_text.split(">", 1)
            if len(parts) > 1 and len(parts[0]) <= 5 and parts[0].endswith(">") and parts[0].startswith("<"):
                generated_text = parts[1].lstrip()
        return generated_text

    def extract_text(self, image_path: str) -> str:
        if not self.model:
            return "Error: Model not initialized."
        if not self.processor and not (self.tokenizer and self.feature_extractor):
            return "Error: Processor/Tokenizer/FeatureExtractor not initialized."

        try:
            image = Image.open(image_path).convert("RGB")

            if self.processor:
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elif self.tokenizer and self.feature_extractor:
                pixel_values = self._process_image_manual(image)
                generated_ids = self.model.generate(pixel_values)
                generated_text = self._decode_text_manual(generated_ids)
            else:
                return "Error: OCR components not properly loaded."

            return generated_text

        except FileNotFoundError:
            return f"Error: Image file not found at {image_path}"
        except Exception as e:
            return f"Error during OCR processing: {e}"

if __name__ == "__main__":
    print("Initializing OCR pipeline for a test run...")

    sample_image_path = "ocr/test_images/isha_upanishad_sample.jpg"
    fallback_dummy_image_path = "ocr_dummy_test_image.png"

    if not os.path.exists(sample_image_path):
        print(f"WARNING: Sample Indic image not found at {sample_image_path}.")
        print(f"Attempting to create and use a dummy English image: {fallback_dummy_image_path}")
        try:
            dummy_image = Image.new('RGB', (600, 100), color='white')
            draw = ImageDraw.Draw(dummy_image)
            draw.text((10, 10), "This is a dummy test image with English text.", fill="black")
            dummy_image.save(fallback_dummy_image_path)
            print(f"Created dummy image: {fallback_dummy_image_path}")
            current_test_image_path = fallback_dummy_image_path
            is_dummy = True
        except ImportError:
            print("Pillow (Image, ImageDraw) not available. Cannot create dummy image.")
            current_test_image_path = None
            is_dummy = False
        except Exception as e_dummy:
            print(f"Could not create dummy image: {e_dummy}")
            current_test_image_path = None
            is_dummy = False
    else:
        print(f"Using sample Indic image: {sample_image_path}")
        current_test_image_path = sample_image_path
        is_dummy = False

    if current_test_image_path and os.path.exists(current_test_image_path):
        ocr_pipeline = IndicOCR()
        if ocr_pipeline.model:
            print(f"\nAttempting OCR on the image: {current_test_image_path}")
            extracted_text = ocr_pipeline.extract_text(current_test_image_path)
            print(f"\n--- Extracted Text ({os.path.basename(current_test_image_path)}) ---")
            print(extracted_text)
            print("--- End of Extracted Text ---")

            if is_dummy:
                print("\nNote: The above text was extracted from a DUMMY ENGLISH image.")
                print("Performance on this dummy image is NOT representative of its Indic OCR capabilities.")
            else:
                print("\nThis text was extracted from an Indic script sample. Please review for accuracy.")
        else:
            print("OCR pipeline model failed to initialize. Skipping OCR test run.")
    elif not current_test_image_path:
        print("No image available to test the OCR pipeline.")
    else:
        print(f"ERROR: Test image path '{current_test_image_path}' was set but file does not exist. Cannot run OCR.")

    print("\nOCR Pipeline script execution finished.")
