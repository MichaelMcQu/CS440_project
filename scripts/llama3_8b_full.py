import os
import json
import requests
from pathlib import Path
from datetime import datetime
import kagglehub
import random

class COTTestPipeline:
    """Pipeline for testing chain-of-thought reasoning in LLMs"""
    
    def __init__(self, model_name="llama3", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.results_dir = Path("cot_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_math_problems_by_category(self, dataset_path, categories, problems_per_category=30):
        """
        Load math problems evenly distributed across categories.
        Returns a list of problems with their category labels.
        """
        all_problems = []
        dataset_path = Path(dataset_path)
        
        for category in categories:
            category_path = dataset_path / "MATH" / "train" / category
            
            if not category_path.exists():
                print(f"Warning: Category path {category_path} does not exist")
                continue
            
            # Get all JSON files in this category
            json_files = list(category_path.glob("*.json"))
            
            if len(json_files) < problems_per_category:
                print(f"Warning: Only {len(json_files)} problems found in {category}, requested {problems_per_category}")
                selected_files = json_files
            else:
                # Randomly sample the requested number
                selected_files = random.sample(json_files, problems_per_category)
            
            print(f"Loading {len(selected_files)} problems from {category}...")
            
            for json_file in selected_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_problems.append({
                            'id': json_file.stem,
                            'category': category,
                            'problem': data.get('problem', ''),
                            'solution': data.get('solution', ''),
                            'level': data.get('level', 'unknown'),
                            'type': data.get('type', 'unknown'),
                            'source_file': str(json_file)
                        })
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        return all_problems
    
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
                timeout=180
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
    
    def run_structured_study(self, dataset_path, categories, temperatures, 
                            problems_per_temp_category=10, checkpoint_interval=15):
        """
        Run a structured study with multiple temperatures and categories.
        
        Args:
            dataset_path: Path to the MATH dataset
            categories: List of category names to test
            temperatures: List of temperature values to test
            problems_per_temp_category: Number of problems per temperature per category
            checkpoint_interval: Save progress every N problems (default: 5)
        """
        total_problems = len(categories) * len(temperatures) * problems_per_temp_category
        
        # Generate timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = self.results_dir / f"checkpoint_{self.model_name}_{self.run_timestamp}.json"
        
        # Check for existing checkpoint
        all_results = []
        completed_ids = set()
        if self.checkpoint_file.exists():
            print(f"Found existing checkpoint: {self.checkpoint_file}")
            response = input("Resume from checkpoint? (y/n): ")
            if response.lower() == 'y':
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get('results', [])
                    completed_ids = set(r['problem_id'] for r in all_results)
                print(f"Resuming: {len(all_results)} problems already completed\n")
        
        print(f"\n{'='*70}")
        print(f"STRUCTURED CHAIN-OF-THOUGHT STUDY")
        print(f"{'='*70}")
        print(f"Categories: {', '.join(categories)}")
        print(f"Temperatures: {temperatures}")
        print(f"Problems per temperature per category: {problems_per_temp_category}")
        print(f"Total problems: {total_problems}")
        print(f"Already completed: {len(all_results)}")
        print(f"Model: {self.model_name}")
        print(f"Checkpoint file: {self.checkpoint_file}")
        print(f"{'='*70}\n")
        
        # Load all problems for each category
        problems_per_category = len(temperatures) * problems_per_temp_category
        all_problems = self.load_math_problems_by_category(
            dataset_path, 
            categories, 
            problems_per_category
        )
        
        if not all_problems:
            print("No problems loaded. Check your dataset path and categories.")
            return
        
        # Organize problems by category
        problems_by_category = {cat: [] for cat in categories}
        for problem in all_problems:
            problems_by_category[problem['category']].append(problem)
        
        # Run tests
        problem_counter = len(all_results)
        
        for category in categories:
            category_problems = problems_by_category[category]
            
            # Split problems for this category across temperatures
            problems_per_temp = len(category_problems) // len(temperatures)
            
            for temp_idx, temperature in enumerate(temperatures):
                start_idx = temp_idx * problems_per_temp
                end_idx = start_idx + problems_per_temp
                temp_problems = category_problems[start_idx:end_idx]
                
                print(f"\n{'─'*70}")
                print(f"Testing: {category.upper()} | Temperature: {temperature}")
                print(f"{'─'*70}")
                
                for i, problem in enumerate(temp_problems, 1):
                    # Skip if already completed
                    if problem['id'] in completed_ids:
                        print(f"Skipping {problem['id']} (already completed)")
                        continue
                    
                    problem_counter += 1
                    print(f"\n[{problem_counter}/{total_problems}] Problem: {problem['id']}")
                    print(f"Category: {category} | Temp: {temperature}")
                    print(f"Question: {problem['problem'][:100]}...")
                    
                    # Query the model
                    response = self.query_ollama(problem['problem'], temperature)
                    
                    if response['success']:
                        extracted_answer = self.extract_final_answer(response['reasoning'])
                        
                        result = {
                            'problem_id': problem['id'],
                            'category': category,
                            'temperature': temperature,
                            'problem': problem['problem'],
                            'expected_solution': problem['solution'],
                            'level': problem['level'],
                            'type': problem['type'],
                            'model_reasoning': response['reasoning'],
                            'extracted_answer': extracted_answer,
                            'model': self.model_name,
                            'timestamp': datetime.now().isoformat(),
                            'source_file': problem['source_file']
                        }
                        
                        print(f"✓ Response received ({len(response['reasoning'])} chars)")
                        print(f"Extracted answer: {extracted_answer[:80]}...")
                    else:
                        result = {
                            'problem_id': problem['id'],
                            'category': category,
                            'temperature': temperature,
                            'problem': problem['problem'],
                            'expected_solution': problem['solution'],
                            'level': problem['level'],
                            'type': problem['type'],
                            'error': response['error'],
                            'timestamp': datetime.now().isoformat(),
                            'source_file': problem['source_file']
                        }
                        print(f"✗ Error: {response['error']}")
                    
                    all_results.append(result)
                    
                    # Save checkpoint periodically
                    if len(all_results) % checkpoint_interval == 0:
                        self.save_checkpoint(all_results, categories, temperatures)
                        print(f"💾 Checkpoint saved ({len(all_results)} problems)")
        
        # Save final results
        self.save_study_results(all_results, categories, temperatures)
        
        # Clean up checkpoint file
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print(f"Checkpoint file removed (study complete)")
        
        return all_results
    
    def save_checkpoint(self, results, categories, temperatures):
        """Save intermediate checkpoint"""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'model': self.model_name,
                    'categories': categories,
                    'temperatures': temperatures,
                    'total_problems': len(results),
                    'timestamp': self.run_timestamp,
                    'checkpoint': True
                },
                'results': results
            }, f, indent=2)
    
    def save_study_results(self, results, categories, temperatures):
        """Save study results with structured organization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete JSON
        output_file = self.results_dir / f"cot_study_{self.model_name}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'model': self.model_name,
                    'categories': categories,
                    'temperatures': temperatures,
                    'total_problems': len(results),
                    'timestamp': timestamp
                },
                'results': results
            }, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_file}")
        print(f"Total problems tested: {len(results)}")
        
        # Generate summary statistics
        self.print_summary(results, categories, temperatures)
        
        # Create human-readable version
        readable_file = self.results_dir / f"cot_study_{self.model_name}_{timestamp}_readable.txt"
        self.save_readable_study_results(results, readable_file, categories, temperatures)
        print(f"Readable version saved to: {readable_file}")
        print(f"{'='*70}")
    
    def print_summary(self, results, categories, temperatures):
        """Print summary statistics"""
        print(f"\nSUMMARY:")
        print(f"  Categories tested: {len(categories)}")
        print(f"  Temperatures tested: {len(temperatures)}")
        
        # Count by category
        print(f"\n  Problems by category:")
        for category in categories:
            count = sum(1 for r in results if r.get('category') == category)
            print(f"    {category}: {count}")
        
        # Count by temperature
        print(f"\n  Problems by temperature:")
        for temp in temperatures:
            count = sum(1 for r in results if r.get('temperature') == temp)
            print(f"    {temp}: {count}")
        
        # Count errors
        errors = sum(1 for r in results if 'error' in r)
        if errors > 0:
            print(f"\n  ⚠ Errors encountered: {errors}")
    
    def save_readable_study_results(self, results, output_file, categories, temperatures):
        """Save results in a human-readable format for manual verification"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("CHAIN-OF-THOUGHT STRUCTURED STUDY RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Categories: {', '.join(categories)}\n")
            f.write(f"Temperatures: {temperatures}\n")
            f.write(f"Total Problems: {len(results)}\n")
            f.write("="*80 + "\n\n")
            
            # Organize by category and temperature
            for category in categories:
                f.write(f"\n{'#'*80}\n")
                f.write(f"CATEGORY: {category.upper()}\n")
                f.write(f"{'#'*80}\n\n")
                
                for temperature in temperatures:
                    category_temp_results = [
                        r for r in results 
                        if r.get('category') == category and r.get('temperature') == temperature
                    ]
                    
                    if not category_temp_results:
                        continue
                    
                    f.write(f"\n{'─'*80}\n")
                    f.write(f"Temperature: {temperature}\n")
                    f.write(f"{'─'*80}\n\n")
                    
                    for i, result in enumerate(category_temp_results, 1):
                        f.write(f"PROBLEM {i}: {result['problem_id']}\n")
                        f.write("-"*80 + "\n")
                        f.write(f"Level: {result.get('level', 'unknown')}\n")
                        f.write(f"Type: {result.get('type', 'unknown')}\n\n")
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
    # Set random seed for reproducibility
    random.seed(42)
    
    # Download latest version
    print("Downloading/locating dataset...")
    path = kagglehub.dataset_download("awsaf49/math-dataset")
    print(f"Path to dataset files: {path}\n")

    # Initialize pipeline
    pipeline = COTTestPipeline(model_name="llama3")
    
    # Define study parameters
    categories = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory"
    ]
    
    # Three temperature values to test
    temperatures = [0.3, 0.7, 1.0]
    
    # Run structured study
    # 5 categories × 3 temperatures × 10 problems = 150 total problems
    results = pipeline.run_structured_study(
        dataset_path=path,
        categories=categories,
        temperatures=temperatures,
        problems_per_temp_category=10
    )
    
    print("\n✓ Study complete! Check the cot_results directory for outputs.")
