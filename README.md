# Bayesian-claim-truthfulness-project
probability reasoning project 

# Bayesian claim truthfulness (Naive Bayes, no deep learning)

Question: how does each piece of evidence (speaker history, context, claim wording)
change the probability that a claim is true? Data: Sentimental LIAR.

## Setup
1. Download the data: https://github.com/UNHSAILLab/SentimentalLIAR
   Copy train_final.csv, valid_final.csv, test_final.csv into a folder called `data`
   in the project folder (it is not stored in this repo).
2. python -m venv .venv, then activate it
   (Windows: .venv\Scripts\activate)
3. pip install -r requirements.txt
4. Open code/truthfulness_full_A.ipynb, select the .venv kernel, click Run All.

## Folders
- code/     notebook and script (truthfulness_nb.py has the same model as readable functions)
- shared/   evidence files for teammates (labels_*.csv, folds.csv, *_llr.csv, prior.txt)
- outputs/  result tables and figures

## Team split
- A: data, plain Naive Bayes, leak fix, threshold and abstain band (Romano)
- B: feature-type ensemble and selection (Peretz)
- C: quantile densities and variable test (Redivo)
