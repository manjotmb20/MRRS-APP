# Recipe Search and Translation Project

## Overview
The Recipe Search and Translation Project is designed to process culinary queries in multiple languages, identifying key ingredients and cooking methods. It can detect and translate queries in English, Spanish, and Hindi, making it a versatile tool for a global audience. The project leverages advanced NLP techniques for parsing and machine learning models for recipe recommendations.

[img](

## Features
- **Multilingual Support:** English, Spanish, Hindi.
- **Query Parsing:** Extracts key information from natural language queries.
- **Translation:** Translates queries across supported languages for broader accessibility.
- **Recipe Search:** Provides relevant recipe suggestions based on query analysis.
- **Semantic Similarity Scoring:** Rates recipes based on relevance to the query.

## Installation

### Prerequisites
Ensure you have Python 3.8+ and pip installed on your system.

### Dependencies
Install the required libraries using pip:

```bash
pip install nltk spacy tensorflow torch langdetect sentence_transformers faiss-cpu googletrans==4.0.0-rc1 stanza
```


## Usage

### Initialization:
Run the `recipe_app.py` script to initialize the application. This sets up the necessary NLP models and embeddings.

### Query Processing:
Input your culinary query into the application. The script will detect the language, parse the query, and translate it if necessary.

### Recipe Search:
Based on the processed query, the application will search through a database of recipes and return the most relevant results.

### View Results:
The application outputs a list of recipes, each with a relevance score indicating its match with the query.

## Contributing
Contributions to the project are welcome! Please adhere to the project's code of conduct and submit pull requests for any enhancements.

## License
MIT License

## Acknowledgements
- Spacy for NLP capabilities.
- Sentence Transformers for semantic text embeddings.
- Google Translate API for translation features.
- Stanza for advanced Hindi language processing.
