import os
import json
import requests
from pathlib import Path
from datetime import datetime
import kagglehub

class COTTestPipeline:
    """Pipeline for testing chain-of-thought reasoning in LLMs"""
    
    def __init__(self, model_name="llama3", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.results_dir = Path("cot_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_math_problems(self, dataset_path, num_problems=10):
        """
        Load math problems from the dataset.
        Expects JSON files with 'problem' and 'solution' fields.
        """
        problems = []
        dataset_path = Path(dataset_path)
        
        # Look for JSON files in the dataset
        json_files = list(dataset_path.rglob("*.json"))
        
        for json_file in json_files[:num_problems]:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    problems.append({
                        'id': json_file.stem,
                        'problem': data.get('problem', ''),
                        'solution': data.get('solution', ''),
                        'source_file': str(json_file)
                    })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                
        return problems[:num_problems]
    
    def query_ollama(self, problem_text, temperature=0.7):
        """
        Query Ollama API with a math problem.
        Returns the full response including reasoning steps.
        """
        prompt = f"""Solve this math problem step by step. Show all your work and reasoning clearly.

Problem: {problem_text}

Please:
1. Break down the problem
2. Show each step of your solution
3. Provide your final answer

Solution:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'reasoning': result.get('response', ''),
                    'model': self.model_name
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def extract_final_answer(self, reasoning_text):
        """
        Attempt to extract the final answer from reasoning text.
        This is a simple heuristic - you may want to customize it.
        """
        lines = reasoning_text.strip().split('\n')
        
        # Look for common answer patterns
        answer_keywords = ['final answer:', 'answer:', 'therefore:', 'thus:']
        
        for line in reversed(lines):
            line_lower = line.lower()
            for keyword in answer_keywords:
                if keyword in line_lower:
                    return line.strip()
        
        # If no keyword found, return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return "Unable to extract answer"
    
    def run_test(self, dataset_path, num_problems=10, temperature=0.7):
        """
        Run the full test pipeline on a subset of problems.
        """
        print(f"Loading {num_problems} problems from {dataset_path}...")
        problems = self.load_math_problems(dataset_path, num_problems)
        
        if not problems:
            print("No problems loaded. Check your dataset path.")
            return
        
        print(f"Loaded {len(problems)} problems")
        print(f"Testing with model: {self.model_name}\n")
        
        results = []
        
        for i, problem in enumerate(problems, 1):
            print(f"Problem {i}/{len(problems)}: {problem['id']}")
            print(f"Question: {problem['problem'][:100]}...")
            
            # Query the model
            response = self.query_ollama(problem['problem'], temperature)
            
            if response['success']:
                extracted_answer = self.extract_final_answer(response['reasoning'])
                
                result = {
                    'problem_id': problem['id'],
                    'problem': problem['problem'],
                    'expected_solution': problem['solution'],
                    'model_reasoning': response['reasoning'],
                    'extracted_answer': extracted_answer,
                    'model': self.model_name,
                    'timestamp': datetime.now().isoformat(),
                    'source_file': problem['source_file']
                }
                
                print(f"✓ Response received ({len(response['reasoning'])} chars)")
                print(f"Extracted answer: {extracted_answer[:80]}...\n")
            else:
                result = {
                    'problem_id': problem['id'],
                    'problem': problem['problem'],
                    'expected_solution': problem['solution'],
                    'error': response['error'],
                    'timestamp': datetime.now().isoformat()
                }
                print(f"✗ Error: {response['error']}\n")
            
            results.append(result)
        
        # Save results
        self.save_results(results)
        return results
    
    def save_results(self, results):
        """Save results to JSON file for manual verification"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"cot_test_{self.model_name}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"Total problems tested: {len(results)}")
        print(f"{'='*60}")
        
        # Also create a human-readable version
        readable_file = self.results_dir / f"cot_test_{self.model_name}_{timestamp}_readable.txt"
        self.save_readable_results(results, readable_file)
        print(f"Readable version saved to: {readable_file}")
    
    def save_readable_results(self, results, output_file):
        """Save results in a human-readable format for manual verification"""
        with open(output_file, 'w') as f:
            f.write("CHAIN-OF-THOUGHT TEST RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"PROBLEM {i}: {result['problem_id']}\n")
                f.write("-"*80 + "\n")
                f.write(f"Question:\n{result['problem']}\n\n")
                
                if 'model_reasoning' in result:
                    f.write(f"Model's Reasoning:\n{result['model_reasoning']}\n\n")
                    f.write(f"Extracted Answer:\n{result['extracted_answer']}\n\n")
                    f.write(f"Expected Solution:\n{result['expected_solution']}\n\n")
                    f.write("Manual Verification:\n")
                    f.write("[ ] Reasoning is correct\n")
                    f.write("[ ] Reasoning has errors but answer is correct\n")
                    f.write("[ ] Both reasoning and answer are incorrect\n")
                    f.write("Notes: ___________________________________________\n\n")
                else:
                    f.write(f"ERROR: {result.get('error', 'Unknown error')}\n\n")
                
                f.write("="*80 + "\n\n")


# Example usage
if __name__ == "__main__":

    # Download latest version
    path = kagglehub.dataset_download("awsaf49/math-dataset")
    print("Path to dataset files:", path)

    # Initialize pipeline
    pipeline = COTTestPipeline(model_name="llama3")
    
    results = pipeline.run_test(
        dataset_path=path,
        num_problems=5,  # Start small for testing
        temperature=0.7
    )
    
    print("\nTest complete! Check the cot_results directory for outputs.")
    