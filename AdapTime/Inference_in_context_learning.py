import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utlis import *
from tqdm import tqdm
from Models import *
from prompt_generation import *
import argparse


os.environ["WANDB_DISABLED"] = "true"

# You can change into local paths if they are downloaded locally
dataset_path = "" 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--CoT', action='store_true', help="Enable Chain-of-Thought (CoT) reasoning.")
    parser.add_argument('--ICL', action='store_true', help="Enable in-context learning during testing.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing results.")
    parser.add_argument('--shorten_story', action='store_true', help="Shorten input stories for models with limited context.")
    parser.add_argument('--print_prompt', action='store_true', help="Print example prompts for inspection.")
    parser.add_argument('--unit_test', action='store_true', help="Run the script in unit test mode with a small subset of data.")
    return parser.parse_args()


def load_test_data(args):
    """Loads the test dataset based on the provided arguments."""
    dataset_selection = ['TGQA', 'TimeQA', 'TimeQA_hard', 'TempReason', 'TempReason_l3'].index(args.dataset)
    dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
    split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
    prefix = ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection]
    
    dataset = load_dataset(dataset_path, f'{dataset_name}_TGR')
    data_test = dataset[prefix + 'test']
    
    if args.unit_test:
        data_test = create_subset(data_test, 10)
    
    return data_test, dataset_name, split_name


def initialize_model_and_tokenizer(model_name):
    """Initializes the model and tokenizer based on the model name."""
    model = None
    tokenizer = None
    if 'Llama' in model_name:
        model_name_cmp = f'meta-llama/{model_name}'
        model_name_cmp = "./Llama-2-7b-hf"
        model_name_cmp = "./DeepSeek-R1-Distill-Llama-8B"
        #model_name_cmp = "./Llama-3.1-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_name_cmp)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            model_name_cmp,  device_map="auto"
        )
        #tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        #model.resize_token_embeddings(len(tokenizer))
        model.eval()
    if 'Qwen' in model_name:
        model_name_cmp = "./Qwen2.5-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_name_cmp, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            model_name_cmp,  device_map="auto",trust_remote_code=True
        )
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        model.eval()
    return model, tokenizer


def generate_and_save_prompts(data_test, model, tokenizer, folder_path, folder_path_tl,args, dataset_name, split_name):
    """Generates prompts, processes them in batches, and saves results."""
    batch_size = 1
    input_prompts = []
    file_paths = []
    samples = []
    
    for i in tqdm(range(len(data_test))):
    #for i in tqdm(range(1000)):
        file_path = f'{folder_path}/{str(i)}.json'
        file_path_tl = f'{folder_path_tl}/{str(i)}.json'
        if os.path.exists(file_path) and (not args.overwrite):
            continue
        if i>1000:
            break
        sample = data_test[i]
        try:
            with open(file_path_tl,'r',encoding='utf-8') as file:
                rule=json.load(file)['prediction']
        except:
            rule="To be determined."
        cur_prompt = my_generate_prompt_ICL(
            dataset_name, split_name, 'CoT' if args.CoT else 'SP', 
            sample['story'], sample['question'], sample['candidates'],#rule,
            args.ICL, args.shorten_story, args.CoT, Q_type=sample['Q-Type']
        )
        #print(cur_prompt)
        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        if len(input_prompts) >= batch_size:
            run_one_batch_ICL(args.model, model, tokenizer, input_prompts, samples, file_paths)
            input_prompts, file_paths, samples = [], [], []

    if len(input_prompts) > 0:
        run_one_batch_ICL(args.model, model, tokenizer, input_prompts, samples, file_paths)

def generate_and_save_prompts_to_json(data_test,  folder_path, args, dataset_name, split_name):
    """Generates prompts, processes them in batches, and saves results."""
    batch_size = 1
    input_prompts = []
    file_paths = []
    samples = []
    story=""
    with open(f'./data_rg/{dataset_name}{split_name}_rg.json','w',encoding='utf-8') as file:
        for i in tqdm(range(len(data_test))):
        #for i in tqdm(range(1000)):
            file_path = f'{folder_path}/{str(i)}.json'
            #file_path_tl = f'{folder_path_tl}/{str(i)}.json'
            sample = data_test[i]
            """
            if story==sample['story']:
                continue
            cur_prompt = my_generate_prompt_TG_trans(
            dataset_name, sample['story'], None, None,
            None, None, args.ICL, args.shorten_story,
            False, dataset_name, mode='test',
            eos_token='', prompt_format='plain'
        )
            story=sample['story']
            """
            if i>1000:
                break
            cur_prompt = my_generate_prompt_ICL(
                dataset_name, split_name, 'CoT' if args.CoT else 'SP', 
                sample['story'], sample['question'], sample['candidates'],
                args.ICL, args.shorten_story, args.CoT, Q_type=sample['Q-Type']
            )
        
            prompt={"prompt":cur_prompt}
            file.write(json.dumps(prompt)+"\n")
           

            

def main():
    args = parse_args()

    # Load dataset and initialize variables
    data_test, dataset_name, split_name = load_test_data(args)
    model_selection = ['gpt-3.5', 'gpt-4', 'Llama2-7b', 'Llama2-13b', 'Llama2-70b','Qwen','DeepSeek-Llama3-8b','Llama-3-8b','deepseek-v3'].index(args.model)
    model_name = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf','Qwen','DeepSeek-Llama3-8b','Llama-3-8b','deepseek-v3'][model_selection]
    
    folder_path = f'../results/{dataset_name}_ICL_{{"CoT" if args.CoT else "SP"}}{split_name}_{model_name}'
    #folder_path_tl = f'../results_retrieved_tl/{dataset_name}_ICL_CoT{split_name}_tl_deepseek-r1'
    folder_path_tl ="../results_retrieved_tl/TimeQA_ICL_CoT_easy_tl_deepseek-r1"
    folder_path = f'../results_dynamic/{dataset_name}_ICL_{("CoT" if args.CoT else "SP")}{split_name}_AdapTime_{model_name}'
    folder_path_rule = f'../results_dynamic/{dataset_name}_ICL_{("CoT" if args.CoT else "SP")}{split_name}_rule2_{model_name}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    # Print example prompts if specified
    if args.print_prompt:
        for i in range(5):
            sample = data_test[i]
            prompt = my_generate_prompt_ICL(
                dataset_name, split_name, 'CoT' if args.CoT else 'SP',
                sample['story'], sample['question'], sample['candidates'],
                args.ICL, args.shorten_story, args.CoT, Q_type=sample['Q-Type']
            )
            print(prompt)
            print('===============================')

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(model_name)

    # Generate and save prompts
    generate_and_save_prompts(data_test, model, tokenizer, folder_path, folder_path_rule, args, dataset_name, split_name)
    #generate_and_save_prompts_to_json(data_test,  folder_path, args, dataset_name, split_name)


if __name__ == "__main__":
    main()