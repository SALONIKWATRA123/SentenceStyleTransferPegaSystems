# Assignment 4: Sentence Style Transfer (Formal ↔ Informal)

## Objective
Build a model/pipeline to convert an informal sentence into a formal one and vice versa.

## Models Used
- **T5 Fine-tuned Model**: `prithivida/informal_to_formal_styletransfer` from Hugging Face Transformers for informal-to-formal conversion.
- **gpt-4o-mini**: Use prompting with gpt-4o-mini for sentence style transfer.


## Project Setup & Usage

### 1. Clone the Repository
Clone or download this repository to your local machine.

### 2. Create and Activate a Virtual Environment (Recommended)
Navigate to the project directory and run:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
With the virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install torch transformers openai python-dotenv nltk streamlit
```

### 4. Set Up OpenAI API Key
Create a `.env` file in the project root with the following content:

```
OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Replace `xxxxxxxx...` with your actual OpenAI API key.

### 5. Run the Notebook
Open `sentence_style_transfer.ipynb` in Jupyter or VS Code. Make sure the kernel/interpreter is set to the Python from your `venv` (see VS Code's top-right kernel picker).

Run all cells sequentially. The notebook will:
   - Import libraries
   - Load the T5 model
   - Provide an inference pipeline
   - Evaluate BLEU score
   - Allow you to test custom sentences

### 6. Run the Streamlit Web App
To launch the web interface:

```bash
streamlit run style_transfer_streamlit.py
```

The app will open in your browser at http://localhost:8501

---

## Requirements
- Python 3.8+
- `transformers`, `datasets`, `torch`, `evaluate`, `pandas`, `numpy`

## UI Demo
Run the UI script for a simple web interface:
```bash
python style_transfer_ui.py
```

---
