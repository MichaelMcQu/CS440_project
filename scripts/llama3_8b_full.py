import os
import json
import requests
from pathlib import Path
from datetime import datetime
import kagglehub
import random

class Pipeline:
    
    def __init__(self, model_name="llama3", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.results_dir = Path("cot_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_problems(self, dataset_path, categories, problems_per_category=30):
        problems = []
        dataset_path = Path(dataset_path)
        
        for category in categories:
            category_path = dataset_path / "MATH" / "train" / category
            
            json_files = list(category_path.glob("*.json"))
            selected_files = random.sample(json_files, problems_per_category)
            
            print(f"loading {len(selected_files)} {category} problems")
            
            for json_file in selected_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        problems.append({
                            'id': json_file.stem,
                            'category': category,
                            'problem': data.get('problem', ''),
                            'solution': data.get('solution', ''),
                            'level': data.get('level', 'unknown'),
                            'type': data.get('type', 'unknown'),
                            'source_file': str(json_file)
                        })
                except Exception as e:
                    print(e)
        
        return problems
    
    def query_ollama(self, problem_text, temperature=0.7):

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
    
    def run_study(self, dataset_path, categories, temperatures, 
                            problems_per_temp=10, checkpoint_interval=15):

        num_problems = len(categories) * len(temperatures) * problems_per_temp
        
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = self.results_dir / f"checkpoint_{self.model_name}_{self.run_timestamp}.json"
        
        # check  for checkpoint
        all_results = []
        completed_ids = set()
        if self.checkpoint_file.exists():
            print(f"Checkpoint: {self.checkpoint_file}")
            response = input("Resume from here? (y/n): ")
            if response.lower() == 'y':
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get('results', [])
                    completed_ids = set(r['problem_id'] for r in all_results)
                print(f"Resuming. {len(all_results)} problems already done\n")
        

        problems_per_category = len(temperatures) * problems_per_temp
        all_problems = self.load_problems(
            dataset_path, 
            categories, 
            problems_per_category
        )
        
        if not all_problems:
            print("NO PROBLEMS LOADED -----------------------------------------------------------------------")
            return
        
        problems_by_category = {cat: [] for cat in categories}
        for problem in all_problems:
            problems_by_category[problem['category']].append(problem)
        
        problem_counter = len(all_results)
        
        for category in categories:
            category_problems = problems_by_category[category]
            
            
            problems_per_temp = len(category_problems) // len(temperatures)
            
            for temp_index, temp in enumerate(temperatures):
                start_index = temp_index * problems_per_temp
                end_index = start_index + problems_per_temp
                temp_problems = category_problems[start_index:end_index]
                
                print("-------------------------------------------------------------")
                print(f"Testing category: {category} with temperature: {temp}")
                print("-------------------------------------------------------------")
                
                for i, problem in enumerate(temp_problems, 1):
                    
                    if problem['id'] in completed_ids:
                        continue
                    
                    problem_counter += 1
                    print(f"\nProblem {problem_counter}/{num_problems}:\n\nProblem: {problem['id']}\nCategory: {category}\nTemp: {temp}")
                    print(f"\nQuestion: {problem['problem'][:100]}")
                    

                    response = self.query_ollama(problem['problem'], temp)
                    
                    if response['success']:
                        
                        result = {
                            'problem_id': problem['id'],
                            'category': category,
                            'temperature': temp,
                            'problem': problem['problem'],
                            'expected_solution': problem['solution'],
                            'level': problem['level'],
                            'type': problem['type'],
                            'model_reasoning': response['reasoning'],
                            'model': self.model_name,
                            'timestamp': datetime.now().isoformat(),
                            'source_file': problem['source_file']
                        }
                        
                        print(f"\nResponse received")
                    else:
                        result = {
                            'problem_id': problem['id'],
                            'category': category,
                            'temperature': temp,
                            'problem': problem['problem'],
                            'expected_solution': problem['solution'],
                            'level': problem['level'],
                            'type': problem['type'],
                            'error': response['error'],
                            'timestamp': datetime.now().isoformat(),
                            'source_file': problem['source_file']
                        }
                        print(f"error: {response['error']}")
                    
                    all_results.append(result)
                    

                    if len(all_results) % checkpoint_interval == 0:
                        self.save_checkpoint(all_results, categories, temperatures)
                        print(f"Checkpoint saved ({len(all_results)} problems)")
        
        self.save_results(all_results, categories, temperatures)
        
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        
        return all_results
    
    def save_checkpoint(self, results, categories, temperatures):

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
    
    def save_results(self, results, categories, temperatures):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_file = self.results_dir / f"{self.model_name}_{timestamp}.json"
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
        
        
        output_file = self.results_dir / f"{self.model_name}_{timestamp}.txt"
        self.save_output_file(results, output_file, categories, temperatures)
        print("Output file saved")
    
    def save_output_file(self, results, output_file, categories, temperatures):

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Categories: {', '.join(categories)}\n")
            f.write(f"Temperatures: {temperatures}\n")
            f.write(f"Number of Problems: {len(results)}\n")
            f.write("================================================================================\n\n")
            
            for category in categories:
                f.write("\n========================================\n")
                f.write(f"Category: {category}\n")
                f.write("========================================\n")

                for temperature in temperatures:
                    category_temp_results = [
                        r for r in results 
                        if r.get('category') == category and r.get('temperature') == temperature
                    ]
                    
                    if not category_temp_results:
                        continue
                    
                    f.write("\n========================================\n")
                    f.write(f"Temperature: {temperature}\n")
                    f.write("========================================\n\n\n")
                    
                    for i, result in enumerate(category_temp_results, 1):
                        f.write(f"Problem {i}: {result['problem_id']}\n")
                        f.write("--------------------------------------------------------------------------------\n")
                        f.write(f"Level: {result.get('level', 'unknown')}\n")
                        f.write(f"Type: {result.get('type', 'unknown')}\n\n")
                        f.write(f"Question:\n{result['problem']}\n\n")
                        
                        if 'model_reasoning' in result:
                            f.write(f"Model's Reasoning:\n{result['model_reasoning']}\n\n")
                            f.write(f"Expected Solution:\n{result['expected_solution']}\n\n")
                        else:
                            f.write(f"ERROR: {result.get('error', 'Unknown error')}\n\n")
                        
                        f.write("\n--------------------------------------------------------------------------------\n")





if __name__ == "__main__":

    random.seed(42)

    path = kagglehub.dataset_download("awsaf49/math-dataset")
    pipeline = Pipeline(model_name="llama3")
    
    categories = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory"
    ]
    
    temperatures = [0.3, 0.7, 1.0]
    

    results = pipeline.run_study(
        dataset_path=path,
        categories=categories,
        temperatures=temperatures,
        problems_per_temp=2
    )
