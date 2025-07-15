# Chatbot Evaluation Folder

This folder contains everything needed to test the educational chatbot's performance.

## Installation

First, install the required packages:

```bash
pip install -r requirements_ragas.txt
```

Or install them individually:
```bash
pip install ragas>=0.1.0 datasets>=2.0.0 pandas>=1.5.0
```

Make sure you have your OpenAI API key set (the script should handle this automatically).

## Quick Start

Just run this from the `ragas/` folder:

```bash
python3 ragas_evaluation.py
```

That's it! The script will test the chatbot and show you the results.

## What's Here

- `ragas_evaluation.py` - The main test script
- `RAGAS_Evaluation_Report.md` - Detailed results and analysis
- `ragas_scores.csv` - Score summary
- `evaluation_dataset.csv` - All test questions and answers
- `requirements_ragas.txt` - Required Python packages

## Latest Test Results

**Most metrics look good, but faithfulness needs work:**

- Faithfulness: 75.0% ❌ (needs to be 80%+)
- Answer Relevancy: 86.9% ✅
- Context Precision: 84.0% ✅  
- Context Recall: 86.7% ✅

## What This Means

The chatbot is pretty good at finding relevant information and giving useful answers to students. It just needs to stick closer to the facts from the source material.

## Files You'll Get

After running the evaluation:
- `ragas_scores.csv` - Quick score summary
- `ragas_summary.json` - Technical details
- `evaluation_dataset.csv` - All test data
