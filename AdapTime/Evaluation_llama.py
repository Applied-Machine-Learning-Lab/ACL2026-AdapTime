import os
import json
import numpy as np
from nltk.tokenize import word_tokenize
import collections
from utlis import *
import argparse





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--SFT', action='store_true', help="whether use SFT")
    parser.add_argument('--ICL_only', action='store_true', help="whether use inference with ICL only")
    parser.add_argument('--CoT', action='store_true', help="whether use CoT")
    parser.add_argument('--RAG', action='store_true', help="whether use RAG")
    parser.add_argument('--ppl', action='store_true', help="whether use perplexity")
    parser.add_argument('--no_TG', action='store_true', help="whether to use the temporal graph or original story as context")
    parser.add_argument('--prompt_format', type=str, default='plain', help="whether use plain (text) or json as prompt format")
    parser.add_argument('--type', type=str, default='', help="method")
    return parser.parse_args()


def calculate_EM(a_gold, a_pred):
    """Calculate Exact Match (EM) score"""
    return a_gold.replace(' ', '').lower() == a_pred.replace(' ', '').lower()


def calculate_F1(a_gold, a_pred):
    """Calculate token-level F1 score"""
    gold_toks = word_tokenize(a_gold)
    pred_toks = word_tokenize(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def parse_generation(pred):
    """Parse generated answers based on rules"""
    for start_identifier in ['Answer:', 'answer is']:
        if start_identifier in pred:
            pred = pred.split(start_identifier)[-1].strip()
            break

    for end_identifier in ['Test:']:
        if end_identifier in pred:
            pred = pred.split(end_identifier)[0].strip()
        break

    if '\n' in pred:
        pred = pred.split('\n')[-1].strip()

    if '(' in pred:
        pred = pred[len(pred.split('(')[0]) + 1:]
    if ')' in pred:
        pred = pred[:- (len(pred.split(')')[-1]) + 1)]

    if len(pred) > 0 and pred[-1] in [')', '.']:
        pred = pred[:-1]

    return pred.strip()


def initialize_metrics(num_question_cat):
    """Initialize dictionaries to store metrics"""
    EM_dict = {i: [0, 0] for i in range(num_question_cat)}
    f1_score_dict = {i: [] for i in range(num_question_cat)}
    recall_dict = {i: [0, 0] for i in range(num_question_cat)}
    return EM_dict, f1_score_dict, recall_dict

import re
def process_file(data, prediction, f_SFT, f_ppl, prompt_format):
    """Process a single file for predictions and metrics calculation"""
    pred = prediction['prediction'].strip()
    #pred = pred.split('\n')[-1]
    if f_SFT:
        if f_ppl:
            pred = data['prediction']
        else:
            _, pred = parse_TGR_pred(pred, prompt_format)
            pred = '' if pred is None else pred
            #print(pred)
            """
            if prompt_format == 'json':
                pred = pred[0]
            """
            if pred:
                pred = re.sub(r'[\s\t]+', ' ', pred.replace("</s>", "")).strip()
            #print(pred)
    else:
        pred = parse_generation(pred)
    
    gts = data['answer']
    gts = [gt[1:-1].strip() if gt[0] == '(' and gt[-1] == ')' else gt for gt in gts]
    return pred, gts


def compute_metrics(folder_path, f_SFT, f_ppl, prompt_format, dataset_name,prediction_file):
    """Compute EM and F1 metrics for all files"""
    num_question_cat = 9 if dataset_name == 'TGQA' else 1
    EM_dict, f1_score_dict,recall_dict = initialize_metrics(num_question_cat)
    
    with open("recall_list.txt", "r") as f:
        recall_list = [float(line.strip()) for line in f]
    with open(prediction_file, 'r', encoding='utf-8') as file:
        prediction = []
        for line in file:
            line = line.strip()
            try:
                prediction.append(json.loads(line))
            except json.JSONDecodeError as e:
                prediction.append(
                    {"prediction": line}  
                )
    #with open(prediction_file, "r") as file:
        #prediction=json.load(file)
     #   prediction = [json.loads(line) for line in file]
    #print(recall_list)
    num_test_samples = 1000
    wr_list=[]
    for i in range(num_test_samples):
        #if recall_list[i]==1:
        #    continue
        file_path = folder_path + f'/{str(i)}.json'
        if not os.path.exists(file_path):
            continue
        with open(file_path) as json_file:
            data = json.load(json_file)
        pred, gts = process_file(data,prediction[i], f_SFT, f_ppl, prompt_format)
        #print("i:",i)
        #print(pred,gts)
        if pred is None:
            continue
                
        if data['Q-Type'] is None:
            data['Q-Type'] = 0

        # For TGQA, Q-Type 2 and 3, we only consider the number in the answer (e.g. 2 year and 2 years are considered the same))
        if dataset_name == 'TGQA' and data['Q-Type'] in [2, 3]:
            pred = pred.split(' ')[0]
            gts = [gt.split(' ')[0] for gt in gts]

        cur_f1_score = [calculate_F1(pred, gt) for gt in gts]
        f1_score_dict[data['Q-Type']].append(max(cur_f1_score))
        
        cur_EM = [calculate_EM(pred, gt) for gt in gts]
        if max(cur_EM)<1:
            wr_list.append(i)
        EM_dict[data['Q-Type']][0] += max(cur_EM)
        EM_dict[data['Q-Type']][1] += 1

        cur_recall = [calculate_recall(pred, gt) for gt in gts]
        recall_dict[data['Q-Type']][0] += max(cur_recall)  
        recall_dict[data['Q-Type']][1] += 1                
    #print(wr_list)
    return EM_dict, f1_score_dict, recall_dict, num_question_cat

def calculate_recall(pred, gt):
    """
     token-level recall
    """
    pred_tokens = set(pred.strip().split())
    gt_tokens = set(gt.strip().split())

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0

    intersection = pred_tokens & gt_tokens
    recall = len(intersection) / len(gt_tokens)
    return recall

def print_results(EM_dict, f1_score_dict, recall_dict, num_question_cat, f_ppl):
    """Print final EM and F1 results"""
    for i in range(num_question_cat):
        if EM_dict[i][1] > 0:
            EM_dict[i][0] = EM_dict[i][0]/EM_dict[i][1]
    print('\nEM:')
    print(np.mean([EM_dict[i][0] for i in range(num_question_cat) if EM_dict[i][1] > 0]), 
          sum(EM_dict[i][1] for i in range(num_question_cat)))
    if not f_ppl:
        print('\nF1 score:')
        print(np.mean([np.mean(f1_score_dict[i]) for i in range(num_question_cat) if len(f1_score_dict[i]) > 0]),
              sum(len(f1_score_dict[i]) for i in range(num_question_cat)))
        print('\nRecall:')
        recall_scores = [recall_dict[i][0] / recall_dict[i][1]
                         for i in range(num_question_cat) if recall_dict[i][1] > 0]
        recall_total = sum(recall_dict[i][1] for i in range(num_question_cat))
        print(np.mean(recall_scores), recall_total)


def main():
    args = parse_args()
    
    dataset_selection = ['TGQA', 'TimeQA', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)
    model_selection = ['gpt-3.5', 'gpt-4', 'Llama2-7b', 'Llama2-13b', 'Llama2-70b','Qwen','DeepSeek-Llama3-8b','Llama-3-8b','deepseek-r1'].index(args.model)

    dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
    split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
    model_name = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf','Qwen','DeepSeek-Llama3-8b','Llama-3-8b','deepseek-r1'][model_selection]
    learning_setting = 'SP' if not args.CoT else 'CoT'

    if args.SFT:
        strategy = 'TGR' if not args.no_TG else 'storyR'
        folder_path = f'../results/{dataset_name}_{strategy}{split_name}'
    else:
        folder_path = f'../results_dynamic/{dataset_name}_ICL_{learning_setting}{split_name}_{model_name}' if args.ICL_only else \
                        f'../results_dynamic/{dataset_name}{split_name}_{model_name}'
        #prediction_file=f'../results_dynamic/data/{dataset_name}{split_name}_{args.type}_output.json'
        prediction_file=f'./data_rg/{dataset_name}{split_name}_{args.type}_output.json'
        #prediction_file=f'data/{dataset_name}{split_name}_{args.type}_output.json'

    if args.ppl:
        folder_path += '_ppl'
    if args.RAG:
        folder_path = f'../results/{dataset_name}_RAG_{learning_setting}{split_name}_{model_name}' if args.ICL_only else \
                        f'../results/{dataset_name}_RAG_{split_name}_{model_name}'

    print(folder_path)
    EM_dict, f1_score_dict, recall_dict, num_question_cat = compute_metrics(folder_path, args.SFT, args.ppl, args.prompt_format, dataset_name,prediction_file)
    print_results(EM_dict, f1_score_dict, recall_dict, num_question_cat, args.ppl)


if __name__ == '__main__':
    main()