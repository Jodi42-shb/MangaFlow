# Manga Translator with Custom YOLO Detection

A powerful manga translation tool that combines custom-trained YOLO models for text bubble detection with OCR and translation capabilities. This project automatically detects text bubbles in manga pages, extracts the text, translates it, and seamlessly inserts the translations back into the original image.

🔥 **Try it out**: [Live Demo on Hugging Face](https://huggingface.co/spaces/ebhon/MangaFlow)

## Features

- **Custom YOLO Detection**: Uses a custom-trained YOLO model specifically designed for manga text bubble detection
- **OCR Integration**: Leverages MangaOCR for accurate text extraction from manga panels
- **Smart Translation**: Implements DeepL translation with manga-specific formatting rules
- **Intelligent Text Insertion**: Automatically sizes and positions translated text to fit within speech bubbles
- **Bulk Processing**: Can process entire folders of manga pages automatically
- **Visual Feedback**: Provides side-by-side comparisons of original and translated pages

## Project Structure

```
manga_translator_project/
├── images/                  # Input manga pages
├── translated_images/       # Output translated pages
├── font/                    # Custom fonts for text insertion
├── manga_translator/        # Core package directory
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration settings
│   ├── detection.py        # YOLO model and text detection
│   ├── image_utils.py      # Image processing utilities
│   ├── ocr.py             # OCR functionality
│   ├── overlay.py         # Text overlay and formatting
│   ├── text_utils.py      # Text processing utilities
│   └── translation.py     # Translation handling
├── main.py                 # Main execution script
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ebhon/MangaFlow.git
cd manga-translator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the custom YOLO model from [YOLO-manga-bubble-detector](https://github.com/ebhon/YOLO-manga-bubble-detector)

4. Set up your DeepL API key:
   - Create a `.env` file in the project root
   - Add your DeepL API key: `DEEPL_API_KEY=your_api_key_here`

## Usage

1. Place your manga pages in the `images` directory
2. Run the main script:
```bash
python main.py
```
3. Find translated pages in the `translated_images` directory

Or try the [online demo](https://huggingface.co/spaces/ebhon/MangaFlow) for single page translation!

## Module Descriptions

- **config.py**: Contains configuration settings and paths
- **detection.py**: Handles YOLO model loading and text region detection
- **image_utils.py**: Provides image processing and enhancement functions
- **ocr.py**: Manages text extraction and validation
- **overlay.py**: Handles text insertion and formatting
- **text_utils.py**: Contains text processing and cleaning functions
- **translation.py**: Manages translation services and post-processing

## Important Note on Translation Accuracy

While this tool provides automated translation capabilities, please note that:
- Translation results may not always be 100% accurate due to the complexity of Japanese language and manga-specific expressions
- Some cultural nuances and wordplay might not be perfectly captured
- The quality of translation depends on the clarity of the original text and the OCR accuracy

We encourage users to:
- Review translations carefully
- Provide feedback on any inaccuracies or improvements
- Report any issues with text detection or translation quality
- Share suggestions for better handling of specific manga styles or expressions

## Custom YOLO Model

This project uses a custom-trained YOLO model specifically designed for manga text bubble detection. The model was trained on a diverse dataset of manga pages to accurately identify different types of text bubbles and speech patterns.

For more information about the YOLO model training and implementation, visit the [YOLO-manga-bubble-detector](https://github.com/ebhon/YOLO-manga-bubble-detector) repository.

## Technical Details

- **Text Detection**: Custom YOLO model trained on manga-specific datasets
- **OCR**: MangaOCR for Japanese text extraction
- **Translation**: DeepL API integration with manga-specific formatting rules
- **Text Insertion**: Custom font handling and intelligent text sizing
- **Image Processing**: OpenCV and PIL for image manipulation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. We especially encourage:
- Feedback on translation quality
- Suggestions for improving text detection
- Reports of any issues or bugs
- Ideas for new features or improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- **Handwitanto Abraham** - [LinkedIn](https://www.linkedin.com/in/handwitanto-abraham/)

## Links

- [Live Demo](https://huggingface.co/spaces/ebhon/MangaFlow)
- [YOLO Model Repository](https://github.com/ebhon/YOLO-manga-bubble-detector)

## Acknowledgments

- MangaOCR for text extraction
- DeepL for translation services

## Deployment

The project is also available as a web application on Hugging Face Spaces. To deploy your own version: