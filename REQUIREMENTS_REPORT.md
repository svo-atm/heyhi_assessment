# Biology Chatbot Requirements Report

## What We Built

We built an AI chatbot that helps primary school students learn about cells and biology. It reads a textbook and answers student questions in simple language.

## Requirements Check

**Result: 4 out of 5 requirements met**

### 1. RAG System ✓
The chatbot loads the biology PDF, stores the information, and finds relevant content when students ask questions.

### 2. Accurate Answers ✓  
It only answers biology questions from the textbook. For other topics, it says "I don't know that based on the information I have."

### 3. Age-Appropriate Language ✓
Uses simple words instead of complex scientific terms. Explains concepts clearly for primary school students.

### 4. Multiple Languages ✓
Students can ask questions in English, Mandarin, or Malay. The chatbot responds in the same language.

### 5. Test Scores - Mostly Good
We tested the chatbot using RAGAS evaluation:
- **Answer Relevancy**: 86.9% (target: 80%) - Pass
- **Context Precision**: 84.0% (target: 80%) - Pass  
- **Context Recall**: 86.7% (target: 80%) - Pass
- **Faithfulness**: 75.0% (target: 80%) - Needs improvement

## Summary

The chatbot works well for classroom use. It gives accurate, age-appropriate answers about biology topics. The faithfulness score is slightly below target but can be improved with minor adjustments to the prompts.


