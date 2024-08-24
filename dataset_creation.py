from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import numpy as np

hf_token = ''  # Add your Hugging Face token here

def load_and_process_datasets(dataset_name: str, config_name: str = None, subset_size: int = 128, text_field: str = 'text', split: str = 'train', isCodeDataSet = False, isProofDataSet = False, text_field1: str = None, text_field2: str = None, isLean = False, isCoqGym = False, is_split=False) -> list:
    """Loads a subset of a dataset from Hugging Face and processes it."""
    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split, streaming=True, use_auth_token=hf_token)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True, use_auth_token=hf_token)
    text_data = []
    dataset_sources = []  # To keep track of dataset sources
    informal_statements = []
    formal_statements = []  
    for i, item in enumerate(dataset):
        if i >= subset_size:
            break
        if isProofDataSet:
            informal_statements.append(f'informal statement {item[text_field]}')
            formal_statements.append(f'formal statement {item[text_field2]}')
            text = f'informal statement {item[text_field]}formal statement {item[text_field2]}'            
            text_data.append(text)
        elif isCodeDataSet:
            text = f'Code {item[text_field1]} Docstring {item[text_field2]}'
            text_data.append(text)  
        elif isLean:
            if item[text_field]:  # Ensure the list is not empty
                # Concatenate the 'state_before', 'state_after', and 'tactic' from traced_tactics
                tactics_data = " ".join([f"State Before: {tactic['state_before']} State After: {tactic['state_after']} Tactic: {tactic['tactic']}" for tactic in item[text_field]])
                text = tactics_data
                text_data.append(text)
            else:
                continue  # Skip if the list is empty
        elif isCoqGym:
            prev_tactics_text = " ".join(item['prev_tactics']) if item['prev_tactics'] else ""
            context_text = " ".join([f"{key}: {value}" for key, value in item['context'].items()])
            text = f"{prev_tactics_text} {context_text}"
            text_data.append(text)
        else:
            text_data.append(item[text_field])
        
        # Keep track of which dataset each sequence came from
        dataset_sources.append(dataset_name)

    if is_split:
        # Shift formal statements by one
        formal_statements = formal_statements[1:] + formal_statements[:1]
        text_data = [inf + formal for inf, formal in zip(informal_statements, formal_statements)]
        dataset_sources = dataset_sources[1:] + dataset_sources[:1]  # Shift dataset sources as well

    return text_data, dataset_sources

# Load and mix datasets
mixed_data = []
dataset_sources = []

# data_AF, sources = load_and_process_datasets('UDACA/AF', None, 1000, text_field='text') 
# mixed_data.extend(data_AF)
# dataset_sources.extend(sources)

# data_c4, sources = load_and_process_datasets('allenai/c4', 'en', 1000)
# mixed_data.extend(data_c4)
# dataset_sources.extend(sources)

# data_dojo, sources = load_and_process_datasets('tasksource/leandojo', None, 1000, split="train", text_field='traced_tactics', isLean=True, isProofDataSet=False)
# mixed_data.extend(data_dojo)
# dataset_sources.extend(sources)

# data_PROOFNET_split, sources = load_and_process_datasets('hoskinson-center/proofnet', None, 186, text_field='nl_statement', text_field2='formal_statement', split = 'validation', isProofDataSet=True, is_split=True) 
# mixed_data.extend(data_PROOFNET_split)
# dataset_sources.extend(sources)

# data_PROOFNET, sources = load_and_process_datasets('hoskinson-center/proofnet', None, 186, text_field='nl_statement', text_field2='formal_statement', split = 'validation', isProofDataSet=True) 
# mixed_data.extend(data_PROOFNET)
# dataset_sources.extend(sources)

# data_AF_SPLIT, sources = load_and_process_datasets('UDACA/AF-split', None, 284, text_field='Statement:')
# mixed_data.extend(data_AF_SPLIT)
# dataset_sources.extend(sources)

# data_wikitext, sources = load_and_process_datasets('wikitext', 'wikitext-103-v1', 428)
# mixed_data.extend(data_wikitext)
# dataset_sources.extend(sources)

# Adding samples from ProofPile 2 dataset subsets
proofpile_subsets = [
    "Agda", "C", "C++", "Coq", "Fortran", "GAP", "Haskell", "Idris", 
    "Isabelle", "Isa_Proofsteps", "Julia", "Jupyter", "Lean", "Lean_Proofsteps", 
    "Maple", "Matlab", "Python", "R", "Tex", 
]

# Load a small sample from the AlgebraicStack to inspect
dataset = load_dataset("EleutherAI/proof-pile-2", "Python", split="train")

# Print out the first few examples to inspect the structure
for i in range(5):
    print(dataset[i])
    print("------------------------")
    
# for subset in proofpile_subsets:
#     dataset = load_dataset("EleutherAI/proof-pile-2", subset, split="train")
#     dataset_size = len(dataset)
#     print(subset)

#     # Determine the number of samples to select
#     num_samples = min(dataset_size, 5000)

#     sampled_subset = dataset.shuffle(seed=42).select(range(num_samples))
#     df_subset = pd.DataFrame(sampled_subset['text'], columns=["text"])
#     df_subset['source'] = f'ProofPile_{subset}'  # Prefix with ProofPile
#     mixed_data.extend(df_subset['text'].tolist())
#     dataset_sources.extend(df_subset['source'].tolist())

# # Combine the mixed data and sources into a DataFrame
# combined_data = pd.DataFrame({
#     'text': mixed_data,
#     'source': dataset_sources
# })

# # Save the combined dataset to a CSV file
# combined_data.to_csv('combined_mixed_dataset.csv', index=False)

# print("Combined dataset saved to 'combined_mixed_dataset.csv'.")