import json
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Run garbage collection
gc.collect()

# Empty PyTorch cache
torch.cuda.empty_cache()

# Optionally reset cached memory statistics
torch.cuda.reset_peak_memory_stats()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load GPT-2 Small model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)
model.eval()

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

def load_data(file_path):
    """Load the JSON data file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_premise_type(prompt):
    """Extract the premise type from a prompt"""
    premise_types = ["Redefine", "Assess", "Fact Check", "Review", "Validate", "Verify"]
    for premise in premise_types:
        if prompt.startswith(premise + ":"):
            return premise
    return "Unknown"

def generate_response(prompt, max_length=250):
    """Generate a response from GPT-2 for the given prompt"""
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Extract only the generated part (excluding the input)
    generated = outputs[0, inputs.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def analyze_responses(data):
    """Analyze responses for different premise types"""
    premise_types = ["Redefine", "Assess", "Fact Check", "Review", "Validate", "Verify"]
    results = {premise: {"baseline": {"factual": 0, "counterfact": 0}, 
                         "ablated": {"factual": 0, "counterfact": 0}} 
               for premise in premise_types}
    
    total_items = len(data)
    
    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"Processing item {i+1}/{total_items}")
        
        premise = extract_premise_type(item['prompt'])
        if premise == "Unknown":
            continue
            
        target_true = item['target_true'].strip()
        target_new = item['target_new'].strip()
        
        # Baseline: use the original prompt with premise
        baseline_response = generate_response(item['prompt'])
        
        # Check if response contains true or counterfactual information
        if target_true.lower() in baseline_response.lower():
            results[premise]['baseline']['factual'] += 1
        elif target_new.lower() in baseline_response.lower():
            results[premise]['baseline']['counterfact'] += 1
        
        # Ablated: remove the premise prefix
        ablated_prompt = item['prompt'].split(':', 1)[1].strip()
        ablated_response = generate_response(ablated_prompt)
        
        # Check if response contains true or counterfactual information
        if target_true.lower() in ablated_response.lower():
            results[premise]['ablated']['factual'] += 1
        elif target_new.lower() in ablated_response.lower():
            results[premise]['ablated']['counterfact'] += 1
    
    return results

def calculate_percentages(results):
    """Calculate percentages for each premise type"""
    percentages = {}
    
    for premise, data in results.items():
        base_factual = data['baseline']['factual']
        base_counterfact = data['baseline']['counterfact']
        base_total = base_factual + base_counterfact
        base_percent = (base_factual / base_total * 100) if base_total > 0 else 0
        
        ablated_factual = data['ablated']['factual']
        ablated_counterfact = data['ablated']['counterfact']
        ablated_total = ablated_factual + ablated_counterfact
        ablated_percent = (ablated_factual / ablated_total * 100) if ablated_total > 0 else 0
        
        percentages[premise] = {
            'baseline': {
                'factual': base_factual,
                'counterfact': base_counterfact,
                'percent_factual': base_percent
            },
            'ablated': {
                'factual': ablated_factual,
                'counterfact': ablated_counterfact,
                'percent_factual': ablated_percent
            }
        }
    
    return percentages

def create_results_table(percentages):
    """Create a results table in the required format"""
    table_data = []
    premises = ["Redefine", "Assess", "Fact Check", "Review", "Validate", "Verify"]
    
    for premise in premises:
        if premise in percentages:
            base = percentages[premise]['baseline']
            ablated = percentages[premise]['ablated']
            
            table_data.append([
                premise,
                base['factual'], base['counterfact'], f"{base['percent_factual']:.2f}%",
                ablated['factual'], ablated['counterfact'], f"{ablated['percent_factual']:.2f}%"
            ])
    
    # Create DataFrame
    df = pd.DataFrame(table_data, columns=[
        'Premise', 
        '#Factual (Baseline)', '#Counterfact (Baseline)', '%Factual (Baseline)',
        '#Factual (Ablated)', '#Counterfact (Ablated)', '%Factual (Ablated)'
    ])
    
    return df

def visualize_results(percentages):
    """Create visualizations of the results"""
    premises = ["Redefine", "Assess", "Fact Check", "Review", "Validate", "Verify"]
    
    # Prepare data for visualization
    baseline_percents = [percentages[p]['baseline']['percent_factual'] for p in premises]
    ablated_percents = [percentages[p]['ablated']['percent_factual'] for p in premises]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(premises))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], baseline_percents, width, label='Baseline')
    ax.bar([i + width/2 for i in x], ablated_percents, width, label='Ablated')
    
    ax.set_xlabel('Premise Type')
    ax.set_ylabel('% Factual Responses')
    ax.set_title('Factuality of Responses by Premise Type')
    ax.set_xticks(x)
    ax.set_xticklabels(premises)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('premise_comparison.png')
    plt.show()
    
    # Create detailed breakdown
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, premise in enumerate(premises):
        labels = ['Factual', 'Counterfactual']
        baseline_sizes = [
            percentages[premise]['baseline']['factual'],
            percentages[premise]['baseline']['counterfact']
        ]
        ablated_sizes = [
            percentages[premise]['ablated']['factual'],
            percentages[premise]['ablated']['counterfact']
        ]
        
        if sum(baseline_sizes) > 0:
            axes[i].pie(baseline_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[i].set_title(f'{premise} - Baseline')
        else:
            axes[i].text(0.5, 0.5, "No Data", ha='center', va='center')
            axes[i].set_title(f'{premise} - Baseline (Empty)')
        
    plt.tight_layout()
    plt.savefig('premise_breakdown.png')
    plt.show()

def main():
    # Load data
    print("Loading data...")
    data = load_data("./Data/gpt2_with_questions_merged.json")

    # Analyze responses
    print("Analyzing responses for different premise types...")
    results = analyze_responses(data)
    
    # Calculate percentages
    percentages = calculate_percentages(results)
    
    # Create results table
    results_table = create_results_table(percentages)
    
    # Display results
    print("\nResults Table:")
    print(results_table.to_string(index=False))
    
    # Save results to CSV
    results_table.to_csv("premise_analysis_results.csv", index=False)
    print("\nResults saved to premise_analysis_results.csv")
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_results(percentages)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for premise in percentages:
        base_percent = percentages[premise]['baseline']['percent_factual']
        ablated_percent = percentages[premise]['ablated']['percent_factual']
        improvement = ablated_percent - base_percent
        print(f"{premise}: Baseline {base_percent:.2f}% -> Ablated {ablated_percent:.2f}% "
              f"(Improvement: {improvement:+.2f}%)")

if __name__ == "__main__":
    main()
