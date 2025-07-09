# Cultural Artifact Explorer & Multimodal Retrieval System

This project aims to build a system for exploring cultural artifacts through a multimodal approach, incorporating Optical Character Recognition (OCR) for text extraction from images, Natural Language Processing (NLP) for understanding and translating this text, and multimodal retrieval (image-text) to find related artifacts.

## Project Structure

The project is organized into the following main directories:

-   `ocr/`: Contains the Indic OCR pipeline (e.g., using Indic-DOCTR or TrOCR). Responsible for extracting text from images of artifacts.
-   `nlp/`: Houses modules for NLP tasks such as:
    -   Translation (e.g., using IndicTrans2)
    -   Summarization (e.g., using mBART for cultural paragraphs)
    -   Named Entity Recognition (NER, e.g., using IndicBERT or XLM-R)
-   `retrieval/`: Implements multimodal embedding and retrieval functionalities:
    -   Embedding images (e.g., using BLIP or CLIP)
    -   Embedding translated text
    -   Nearest neighbor search using FAISS or Annoy for querying (image-to-text, text-to-image).
-   `interface/`: Contains the frontend application (e.g., built with Streamlit or React) to interact with the system.
-   `utils/`: Provides helper functions for tasks like preprocessing, logging, etc.
-   `tests/`: (To be added) Will contain unit and integration tests for the various modules.
-   `training_configs/`: Contains configurations for training models.
-   `processed_data/`: Stores datasets that have been processed and are ready for model training/evaluation.
-   `datasets/`: Location for raw datasets.

## Core Functionalities (Planned)

-   **OCR Pipeline**:
    -   Load and utilize pretrained Indic-OCR models.
    -   Accept image input and output extracted Indic text.
    -   Visualize text overlaid on images.
-   **NLP Pipeline**:
    -   Translate extracted text (e.g., from Indic languages to English).
    -   Summarize relevant textual content.
    -   Identify named entities within the text.
-   **Multimodal Retrieval**:
    -   Generate embeddings for images and their corresponding textual descriptions.
    -   Build a searchable index of these embeddings.
    -   Allow users to query with an image to find related texts/artifacts or with text to find related images.
-   **User Interface**:
    -   A simple interface (potentially Streamlit) to:
        -   Upload an image to trigger OCR, translation, and summarization.
        -   Enter a text query to retrieve matching artifact images or information.

## Setup and Running

(Instructions to be added as components are developed. This will include dependencies, model downloads, and how to run the application.)

## Placeholder Execution

Currently, most modules contain placeholder scripts. These can be run individually to see print statements simulating their intended behavior. For example:

```bash
python ocr/ocr_pipeline.py
python nlp/translation.py
# etc.
```

If Streamlit is installed, the placeholder UI can potentially be run (after uncommenting its code in `interface/app.py` and ensuring necessary dummy files like `sample_ui_image.png` can be created or exist):

```bash
streamlit run interface/app.py
```
