import os
import argparse
import time
import json
import numpy as np
from tqdm import tqdm
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer


from rouge_score import rouge_scorer
from preprocess import DatasetProcessor
from models import BertClassifier

parser = argparse.ArgumentParser(description='Text Summarization')
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--kernel_num', type=int, default=100)
parser.add_argument('--kernel_sizes', type=str, default='1,2,3')
parser.add_argument('--pretrained_bert', type=str, default='bert-base-uncased')
parser.add_argument('--pretrained_sentence_bert', type=str, default='bert-base-nli-mean-tokens')
# parser.add_argument('--rouge_path', type=str, default='/content/drive/My Drive/TextSumarization/pyrouge/tools/ROUGE-1.5.5/')
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_sequence_length', type=int, default=128)
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--topk', type=int, default=4)
parser.add_argument('--train_path', type=str, default='./data/USAToday-CNN.json')
args = parser.parse_args()



print("ROUGE setup successful!")
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Cuda setup successful!")
else:
    device = torch.device("cpu")
    print("Cpu setup successful!")

def train(model, train_dataloader, epochs=args.epochs):
    print("Start training...\n")
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Elapsed':^9}")
        print("-"*70)

        t0_batch = time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            input_ids, attention_masks, label, comments_feature = tuple(t.to(device) for t in batch)
            model.zero_grad()

            logits = model(input_ids, attention_masks, comments_feature)

            loss = loss = loss_fn(logits, label)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {time_elapsed:^9.2f}")
        avg_train_loss = total_loss / len(train_dataloader)

def read_text_files(directory):
    texts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                texts[filename] = text
                print(f"Reading {filename}:")
                print(text[:100])  # Print the first 100 characters for quick inspection
                print('-' * 50)  # Separator between files
    return texts
def predict(model, test_id, processor, fold, evaluation):
    model.eval()
    os.makedirs(f"evaluation/system_summaries/{fold}", exist_ok=True)
    os.makedirs(f"evaluation/model_summaries/{fold}", exist_ok=True)

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    with torch.no_grad():
        for idx, i in tqdm(enumerate(test_id, 0)):
            predicted_sentences = []
            paragraph = []
            all_comments_emb = processor.get_all_features_of_comments(i)
            
            for sent in processor.data[i]["document"]["sentences"]["sentence"]:
                text = tokenizer.cls_token + processor.data[i]["title"] + tokenizer.sep_token + sent["content"] + tokenizer.sep_token
                input_ids, attention_masks = processor.convert_example_to_feature(text)
                # print(text)
                comment_feature = processor.get_feature_of_best_comment(sent["content"], all_comments_emb)
                logits = model(input_ids.to(device), attention_masks.to(device), comment_feature.to(device))
                score = F.softmax(logits, dim=1)
                predicted_sentences.append((sent["content"].lower(), score[0][1].item()))

            # Sort and limit sentences
            predicted_sentences = sorted(predicted_sentences, key=lambda x: x[1], reverse=True)
            if len(predicted_sentences) > args.topk:
                predicted_sentences = predicted_sentences[:args.topk]

            summaries = [sentence[0] for sentence in predicted_sentences]
            paragraph = [sent["content"].lower() for sent in processor.data[i]["summary"]["sentences"]["sentence"]]

            # Save summaries to files
            with open(f"evaluation/system_summaries/{fold}/text.{idx}.txt", mode="w", encoding="utf-8") as fout_1:
                fout_1.write("\n".join(summaries))
            with open(f"evaluation/model_summaries/{fold}/text.A.{idx}.txt", mode="w", encoding="utf-8") as fout_2:
                fout_2.write("\n".join(paragraph))

    # Load summaries
    model_summaries = read_text_files(f"evaluation/model_summaries/{fold}")
    system_summaries = read_text_files(f"evaluation/system_summaries/{fold}")

    results = {}
    

    model_files = list(model_summaries.keys())
    system_files = list(system_summaries.keys())
    # Match based on index
    for model_file, system_file in zip(model_files, system_files):
        model_text = model_summaries[model_file]
        system_text = system_summaries[system_file]

        # Calculate ROUGE scores
        scores = scorer.score(system_text, model_text)
        results[f'{model_file} vs {system_file}'] = scores

    # Append the results to evaluation
    evaluation.append(results)

    # Print the ROUGE scores
    print(f"ROUGE scores for fold {fold}:")
    pprint(results)

    return evaluation



def initialize_model(epochs=args.epochs):
    bert_classifier = BertClassifier(args)
    bert_classifier.to(device)
    optimizer = AdamW(bert_classifier.parameters(), lr=args.lr, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=50,
                                                num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()
    return bert_classifier, optimizer, scheduler, loss_fn

def evaluate_modelbert(evaluation):
    # Lists to store average metrics for ROUGE-1, ROUGE-2, and ROUGE-L
    rouge_1_f1_scores = []
    rouge_2_f1_scores = []
    rouge_l_f1_scores = []

    # Iterate over each evaluation dictionary
    for eval_result in evaluation:
        for key, rouge_scores in eval_result.items():
            # Extract fmeasure for ROUGE-1, ROUGE-2, and ROUGE-L
            rouge_1_f1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge_2_f1_scores.append(rouge_scores['rouge2'].fmeasure)
            rouge_l_f1_scores.append(rouge_scores['rougeL'].fmeasure)

    # Calculate the mean scores
    avg_rouge_1 = np.mean(rouge_1_f1_scores) if rouge_1_f1_scores else 0
    avg_rouge_2 = np.mean(rouge_2_f1_scores) if rouge_2_f1_scores else 0
    avg_rouge_l = np.mean(rouge_l_f1_scores) if rouge_l_f1_scores else 0

    # Prepare results dictionary
    results = {
        "Average ROUGE-1 F1": round(avg_rouge_1, 4),
        "Average ROUGE-2 F1": round(avg_rouge_2, 4),
        "Average ROUGE-L F1": round(avg_rouge_l, 4),
    }

    # Print results
    print(json.dumps(results, indent=3))

    return results

if __name__ == "__main__":
    
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)
    processor = DatasetProcessor(args, tokenizer)
    kf = KFold(n_splits=args.n_folds)
    i = 0
    evaluation = []
    for train_id, test_id in kf.split(processor.data):
        print(train_id)
        print("Training in fold :", i)
        train_dataloader = processor.load_training_data(train_id)
        bert_classifier, optimizer, scheduler, loss_fn = initialize_model()
        train(bert_classifier, train_dataloader)
        evaluation = predict(bert_classifier, test_id, processor, i, evaluation)
        i += 1
        save_path = f"saved_models/bert_classifier_fold_{i}.pt"
        torch.save(bert_classifier.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        print("Evaluation Shape:", len(evaluation))
        print("Evaluation Type:", type(evaluation))
        evaluate_modelbert(evaluation)
        
    print(evaluation)
    evaluate_modelbert(evaluation)