"""
RAGAS Evaluation for Educational RAG Chatbot

This script evaluates the chatbot using the RAGAS framework to assess:
- Faithfulness (>80%)
- Answer Relevancy (>80%) 
- Context Precision (>80%)
- Context Recall (>80%)

Run: python ragas_evaluation.py
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import asyncio
from dotenv import load_dotenv

# Add parent directory to path for imports
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Change working directory temporarily for data loading
original_dir = os.getcwd()
os.chdir(parent_dir)

# Load environment variables
load_dotenv()

try:
    from ragas.evaluation import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    print("RAGAS libraries imported successfully")
except ImportError as e:
    print(f"RAGAS import issue: {e}")
    try:
        # Try alternative import structure
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        RAGAS_AVAILABLE = True
        print("RAGAS libraries imported with alternative method")
    except ImportError as e2:
        print(f"RAGAS not available: {e2}")
        RAGAS_AVAILABLE = False

# Import chatbot components (now from parent directory)
from chatbot import create_parent_chain, vectorstore, chat_model, embeddings
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

# Return to original directory (ragas/)
os.chdir(original_dir)

class RAGASEvaluator:
    def __init__(self):
        """Initialize the RAGAS evaluator."""
        self.store = {}
        store_path = os.path.join(parent_dir, "store_location")
        self.fs = LocalFileStore(store_path)
        self.parent_store = create_kv_docstore(self.fs)
        self.chain = create_parent_chain(vectorstore, self.store, self.parent_store)
        
        # Set up RAGAS with proper configuration
        if RAGAS_AVAILABLE:
            self.setup_ragas_metrics()

    def setup_ragas_metrics(self):
        """Set up RAGAS metrics with proper LLM configuration."""
        try:
            # Initialize metrics with default configuration
            # RAGAS will use environment variables for OpenAI configuration
            print("RAGAS metrics configured successfully")
        except Exception as e:
            print(f"RAGAS metric configuration issue: {e}")

    def get_test_cases(self) -> List[Dict[str, str]]:
        """Get test cases covering various educational aspects."""
        return [
            {
                "question": "What are cells and why are they called the building blocks of life?",
                "ground_truth": "Cells are the simplest structural and functional units of life. All living things are made up of billions of tiny cells, just as a building is made of bricks. They are called building blocks because they form the basic structure of all living organisms."
            },
            {
                "question": "Who discovered cells and how did they get their name?",
                "ground_truth": "Robert Hooke, an English scientist, first introduced the term 'cells' in 1667. He used early microscopes to examine cork and saw closely packed little boxes with thick walls, which he named 'cells' because they looked like cells in a honeycomb or prison."
            },
            {
                "question": "What is protoplasm and what are its main components?",
                "ground_truth": "Protoplasm is a complex jelly-like substance that makes up the living matter of a cell. It consists of three main parts: the cell membrane (cell surface membrane), cytoplasm, and nucleus. Chemical activities that allow the cell to survive and grow are carried out in the protoplasm."
            },
            {
                "question": "How do light microscopes and electron microscopes differ?",
                "ground_truth": "Light microscopes magnify objects up to 1000 times and can produce color micrographs. Electron microscopes magnify objects to more than 200,000 times and produce black-and-white micrographs that can be artificially colorized."
            },
            {
                "question": "What is cytoplasm and where is it located?",
                "ground_truth": "Cytoplasm is a jelly-like substance that fills the inside of the cell and is enclosed by the cell membrane. It is the part of the protoplasm between the cell membrane and the nucleus, where most cell activities occur."
            },
            {
                "question": "Why are cells compared to chemical factories?",
                "ground_truth": "Cells are like chemical factories because many chemical reactions occur continually inside them to keep organisms alive. They take in raw materials, process them to make new molecules, and these molecules can be used by the cell or transported to other parts of the body."
            },
            {
                "question": "What did Robert Hooke actually observe when he looked at cork?",
                "ground_truth": "Robert Hooke observed closely packed little boxes with thick walls when examining thin slices of cork from tree bark. He only saw the walls of dead plant cells, not living cells themselves."
            },
            {
                "question": "What are micrographs and how are they created?",
                "ground_truth": "Micrographs are pictures taken with microscopes. A camera can be fitted to either light or electron microscopes to take these pictures. Light micrographs can be color images, while electron micrographs are black-and-white but can be artificially colorized."
            },
            {
                "question": "How do different cell structures work together?",
                "ground_truth": "Different cell structures perform different roles within the cell, like departments in a factory. This division of labor increases efficiency and ensures that the cell can survive and perform its role within the body."
            },
            {
                "question": "What is the difference between planets and stars?", # Out-of-scope
                "ground_truth": "This question is outside the scope of the educational content about cells and chemistry of life."
            }
        ]

    def get_response_and_context(self, question: str) -> tuple:
        """Get response and context from the chatbot."""
        try:
            # Get context using retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in docs]
            
            # Get response from chain
            response = self.chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": "ragas_eval_session"}}
            )
            
            return response, contexts
            
        except Exception as e:
            print(f"Error processing '{question}': {e}")
            return f"Error: Unable to process question", ["Error retrieving context"]

    async def run_evaluation(self) -> Dict[str, Any]:
        """Run the RAGAS evaluation."""
        print("Starting RAGAS Evaluation")
        print("=" * 50)
        
        # Get test cases
        test_cases = self.get_test_cases()
        print(f"Created {len(test_cases)} test cases")
        
        # Generate responses and contexts
        evaluation_data = []
        print(f"\nGenerating responses for {len(test_cases)} questions...")
        
        for i, case in enumerate(test_cases, 1):
            question = case["question"]
            print(f"  {i:2d}/{len(test_cases)}: {question[:60]}{'...' if len(question) > 60 else ''}")
            
            answer, contexts = self.get_response_and_context(question)
            
            evaluation_data.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": case["ground_truth"]
            })
        
        # Save evaluation dataset
        df = pd.DataFrame(evaluation_data)
        df.to_csv("evaluation_dataset.csv", index=False)
        print(f"\nSaved evaluation dataset to 'evaluation_dataset.csv'")
        
        if not RAGAS_AVAILABLE:
            print("RAGAS not available - cannot generate scores")
            return {"error": "RAGAS framework not available"}
        
        # Run RAGAS evaluation
        try:
            print("\nRunning RAGAS evaluation...")
            dataset = Dataset.from_list(evaluation_data)
            
            # Use all RAGAS metrics
            metrics_to_use = [faithfulness, answer_relevancy, context_precision, context_recall]
            
            result = evaluate(dataset, metrics=metrics_to_use)
            print(f"RAGAS result type: {type(result)}")
            print(f"RAGAS result: {result}")
            
            # Check if result is valid
            if result is None or result == 0:
                print("RAGAS returned invalid result, creating fallback summary")
                return self.create_fallback_summary(evaluation_data)
            
            return self.process_results(result, evaluation_data)
            
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            return self.create_fallback_summary(evaluation_data)

    def create_fallback_summary(self, evaluation_data: List[Dict]) -> Dict[str, Any]:
        """Create a basic summary when RAGAS fails."""
        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "total_questions": len(evaluation_data),
            "in_scope_questions": len([d for d in evaluation_data if "cell" in d["question"].lower() or "microscope" in d["question"].lower()]),
            "scores_percentage": {},
            "threshold": 80.0,
            "requirements_met": False,
            "evaluation_type": "RAGAS Framework Assessment (Failed)",
            "chatbot_version": "Educational RAG Chatbot v1.0",
            "error": "RAGAS evaluation did not complete successfully"
        }
        
        # Save basic files
        with open("ragas_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\nRAGAS evaluation did not complete successfully.")
        print("Dataset saved, but no metric scores available.")
        print(f"Please check RAGAS installation and OpenAI API access.")
        
        return summary

    def process_results(self, result: Any, evaluation_data: List[Dict]) -> Dict[str, Any]:
        """Process and save RAGAS results."""
        scores = {}
        
        print(f"Processing RAGAS result: {result}")
        print(f"Result type: {type(result)}")
        
        try:
            # RAGAS EvaluationResult has a to_pandas() method that provides metric scores
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                print(f"Result DataFrame shape: {df.shape}")
                print(f"Result DataFrame columns: {df.columns.tolist()}")
                
                # Extract mean scores for each metric
                for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                    if metric in df.columns:
                        raw_score = df[metric].mean()
                        scores[metric] = raw_score * 100  # Convert to percentage
                        print(f"✓ {metric}: {raw_score:.4f} -> {scores[metric]:.1f}%")
                    else:
                        print(f"Warning: {metric} not found in DataFrame columns")
            
            # Alternative: try using the scores attribute
            elif hasattr(result, 'scores') and result.scores:
                print("Using scores attribute...")
                # result.scores is a list of dictionaries
                scores_list = result.scores
                print(f"Scores list: {scores_list}")
                
                if scores_list and len(scores_list) > 0:
                    # Calculate mean scores across all evaluations
                    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                        metric_scores = [score.get(metric, 0) for score in scores_list if metric in score]
                        if metric_scores:
                            raw_score = sum(metric_scores) / len(metric_scores)
                            scores[metric] = raw_score * 100
                            print(f"✓ {metric}: {raw_score:.4f} -> {scores[metric]:.1f}%")
                        else:
                            print(f"Warning: {metric} not found in scores")
            
            else:
                print("Unable to extract scores from result object")
                
        except Exception as e:
            print(f"Error processing RAGAS result: {e}")
            import traceback
            traceback.print_exc()
        
        if not scores:
            print("No scores found, falling back to basic summary")
            return self.create_fallback_summary(evaluation_data)
        
        # Create comprehensive summary
        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "total_questions": len(evaluation_data),
            "in_scope_questions": len([d for d in evaluation_data if "cell" in d["question"].lower() or "microscope" in d["question"].lower()]),
            "scores_percentage": scores,
            "threshold": 80.0,
            "requirements_met": all(score >= 80.0 for score in scores.values()),
            "evaluation_type": "RAGAS Framework Assessment",
            "chatbot_version": "Educational RAG Chatbot v1.0"
        }
        
        # Save files in current directory (ragas/)
        scores_df = pd.DataFrame([scores])
        scores_df.to_csv("ragas_scores.csv", index=False)
        
        with open("ragas_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Display results
        self.display_results(scores, summary)
        
        return summary

    def display_results(self, scores: Dict[str, float], summary: Dict):
        """Display evaluation results."""
        
        print(f"\nRAGAS EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"Date: {summary['evaluation_date'][:19]}")
        print(f"Questions: {summary['total_questions']} total, {summary['in_scope_questions']} in-scope")
        print(f"Threshold: {summary['threshold']}%")
        
        print("\nMETRIC SCORES:")
        print("-" * 40)
        
        for metric, score in scores.items():
            status = "PASS" if score >= 80.0 else "FAIL"
            metric_display = metric.replace("_", " ").title()
            print(f"{metric_display:20} {score:6.1f}% {status}")
        
        print("\n" + "=" * 60)
        
        if summary["requirements_met"]:
            print("OVERALL RESULT: ALL REQUIREMENTS MET")
            print("All metrics scored above 80% threshold")
            print("Chatbot passes RAGAS evaluation criteria")
        else:
            failed_metrics = [m for m, s in scores.items() if s < 80.0]
            print("OVERALL RESULT: REQUIREMENTS NOT MET")
            print(f"Failed metrics: {', '.join(failed_metrics)}")
        
        print(f"\nGenerated Files (in ragas/ directory):")
        print("  • evaluation_dataset.csv - Complete test dataset")
        print("  • ragas_scores.csv - Metric scores")
        print("  • ragas_summary.json - Complete evaluation summary")
        
        print(f"\nRAGAS FRAMEWORK ASSESSMENT COMPLETE")
        print(f"{'PASSED' if summary['requirements_met'] else 'FAILED'} - Educational chatbot evaluation")

async def main():
    """Main execution function."""
    try:
        print("Running RAGAS evaluation from organized directory structure")
        print(f"Working directory: {os.getcwd()}")
        print(f"Parent directory: {parent_dir}")
        
        evaluator = RAGASEvaluator()
        result = await evaluator.run_evaluation()
        
        print(f"\nRAGAS Evaluation Complete")
        
        if result.get("requirements_met", False):
            print("SUCCESS: All RAGAS requirements met (>80% threshold)")
            print("Chatbot is ready for educational use")
        elif "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print("Some requirements not met - review scores above")
            
        return result
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    result = asyncio.run(main())
