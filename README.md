# Stack Overflow Question Quality Classification

This project classifies Stack Overflow questions into three quality labels: HQ, LQ_EDIT, and LQ_CLOSE. It includes a traditional feature-based baseline, an attention-based BiLSTM model, evaluation utilities, attention visualizations, and a Streamlit interface for interactive predictions.

## Dataset

The data is sourced from Kaggle and corresponds to the study cited below. Dataset link: https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate. The training scripts expect CSV files with at least the following columns:

- `Id`: question identifier
- `Title`: question title
- `Body`: question body in Markdown or HTML
- `Y`: label (HQ, LQ_EDIT, LQ_CLOSE)
- `Tags`: required for feature-based baseline
- `CreationDate`: required for feature-based baseline

## Project Structure and File Usage

Root files:

- `requirements.txt`: Python dependencies for training, evaluation, and the Streamlit app.
- `streamlit_app.py`: Streamlit UI for running the attention model on new questions and displaying attention highlights.

Source package:

- `src/__init__.py`: Package marker and module namespace.
- `src/config.py`: Shared constants (labels, label mappings) and artifact directory definitions.
- `src/data.py`: CSV loading, stratified splits, token preparation, dataset and batch utilities.
- `src/preprocessing.py`: Text cleaning, URL and code block normalization, tokenization, and detokenization.
- `src/vocab.py`: Vocabulary building, token to index encoding, and special tokens.
- `src/features.py`: Handcrafted feature extraction for the baseline model.
- `src/train_baseline.py`: Trains a logistic regression baseline and writes metrics, model, and prediction files.
- `src/train_attention.py`: Trains the attention-based BiLSTM, writes checkpoints, metrics, predictions, and a confusion matrix.
- `src/evaluate.py`: Computes metrics from a predictions CSV and optionally saves them as JSON.
- `src/visualize_attention.py`: Generates attention plots for a specific question ID using a saved checkpoint.
- `src/models/attention_lstm.py`: PyTorch definition of the attention-based BiLSTM classifier.

Notes:

- `src/__pycache__/` contains Python bytecode caches generated at runtime and can be ignored.

## Usage

Install dependencies in an environment that you would like:

```bash
pip install -r requirements.txt
```

Train the baseline model:

```bash
python -m src.train_baseline --train_csv train.csv --valid_csv valid.csv
```

Train the attention model:

```bash
python -m src.train_attention --train_csv train.csv --valid_csv valid.csv
```

Evaluate predictions from a CSV:

```bash
python -m src.evaluate --predictions_csv artifacts/metrics/attn_bilstm_predictions.csv
```

Visualize attention for a specific question:

```bash
python -m src.visualize_attention --checkpoint artifacts/checkpoints/attn_bilstm_best.pt --csv valid.csv --question_id 34565049
```

Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Citation

Dataset used was from the paper - Multi-View Approach to Suggest Moderation Actions in Community Question Answering Sites
by 
Annamoradnejad, Issa and Habibi, Jafar and Fazli, Mohammadamin

URL for the paper: https://www.sciencedirect.com/science/article/pii/S0020025522003127
