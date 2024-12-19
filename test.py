# # import os
# # import argparse
# # import time
# # import json
# # import numpy as np
# # from tqdm import tqdm

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# # from sklearn.model_selection import KFold
# # from transformers import AdamW, get_linear_schedule_with_warmup
# # from transformers import BertModel, BertTokenizer

# # from pyrouge import Rouge155
# # from preprocess import DatasetProcessor
# # from models import BertClassifier

# # parser = argparse.ArgumentParser(description='Text Summarization')
# # parser.add_argument('--dropout', type=float, default=0.4)
# # parser.add_argument('--kernel_num', type=int, default=100)
# # parser.add_argument('--kernel_sizes', type=str, default='1,2,3')
# # parser.add_argument('--pretrained_bert', type=str, default='bert-base-uncased')
# # parser.add_argument('--pretrained_sentence_bert', type=str, default='bert-base-nli-mean-tokens')
# # # parser.add_argument('--rouge_path', type=str, default='/content/drive/My Drive/TextSumarization/pyrouge/tools/ROUGE-1.5.5/')
# # parser.add_argument('--lr', type=float, default=3e-5)
# # parser.add_argument('--epochs', type=int, default=1)
# # parser.add_argument('--batch_size', type=int, default=32)
# # parser.add_argument('--max_sequence_length', type=int, default=128)
# # parser.add_argument('--n_folds', type=int, default=5)
# # parser.add_argument('--topk', type=int, default=4)
# # parser.add_argument('--train_path', type=str, default='./data/USAToday-CNN.json')
# # args = parser.parse_args()

# # # r = Rouge155(args.rouge_path)
# # rouge_path = "E:/Web_2024/bert/testt/TextSummarization/pyrouge/tools/ROUGE-1.5.5/"
# # r = Rouge155(rouge_path)
# # print("ROUGE setup successful!")
# # if torch.cuda.is_available():       
# #     device = torch.device("cuda")
# #     print("Cuda setup successful!")
# # else:
# #     device = torch.device("cpu")
# #     print("Cpu setup successful!")


# # def load_trained_model(model_path, args):
# #     model = BertClassifier(args)  # Ensure the model class matches the saved one
# #     model.load_state_dict(torch.load(model_path, map_location=device))
# #     model.to(device)
# #     model.eval()
# #     print("Model loaded successfully!")
# #     return model


# # def predict_single_sample(model, tokenizer, processor, input_text, topk=args.topk):
# #     model.eval()
# #     with torch.no_grad():
# #         all_comments_emb = processor.get_all_features_of_comments(0)  # Dummy index for simplicity
# #         text = tokenizer.cls_token + tokenizer.sep_token + input_text + tokenizer.sep_token
# #         input_ids, attention_masks = processor.convert_example_to_feature(text)
        
# #         # Get comment features for dummy input
# #         comment_feature = processor.get_feature_of_best_comment(input_text, all_comments_emb)
        
# #         # Run through the model
# #         logits = model(input_ids.to(device), attention_masks.to(device), comment_feature.to(device))
# #         score = F.softmax(logits, dim=1)
        
# #         # Output the top predictions
# #         predicted_sentence = [(input_text, score[0][1].item())]
# #         predicted_sentence = sorted(predicted_sentence, key=lambda x: x[1], reverse=True)
# #         return predicted_sentence[:topk]

# # import os
# # os.environ['PATH'] += ';C:/Strawberry/perl/bin'
# # if __name__ == "__main__":
# #     tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)
# #     processor = DatasetProcessor(args, tokenizer)
    
# #     # Load the trained model
# #     model_path = "bert_summarizer.pth"
# #     bert_classifier = load_trained_model(model_path, args)
    
# #     # Test a single input sample
# #     input_sample = "For generating summaries, we make use of an NMT model. The function below loads in data, sends it through that model and formats the summary at the end."
# #     predicted_summary = predict_single_sample(bert_classifier, tokenizer, processor, input_sample)
    
# #     print("Predicted Summary:")
# #     for sentence in predicted_summary:
# #         print(sentence[0])  # Print the predicted sentences
# import os
# from rouge_score import rouge_scorer
# import numpy as np
# def read_text_files(directory):
#     texts = {}
#     for filename in os.listdir(directory):
#         if filename.endswith(".txt"):
#             with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
#                 text = file.read()
#                 texts[filename] = text
#                 print(f"Reading {filename}:")
#                 print(text[:100])  # Print the first 100 characters for quick inspection
#                 print('-' * 50)  # Separator between files
#     return texts

# # Paths to the directories
# model_summaries_path = 'evaluation/model_summaries/0'
# system_summaries_path = 'evaluation/system_summaries/0'

# # Load model and system summaries
# model_summaries = read_text_files(model_summaries_path)
# system_summaries = read_text_files(system_summaries_path)

# # Initialize the ROUGE scorer
# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# print(scorer)
# print("ok")
# # Calculate ROUGE scores for each pair
# results = {}
# model_files = list(model_summaries.keys())
# print(model_files)
# system_files = list(system_summaries.keys())

# # print(model_summaries.items())
# # scores = scorer.score(model_summaries[1], system_summaries[1])

# # for model_file, model_text in model_summaries.items():
# #     print(model_file)
# #     if model_file in system_summaries:
        
# #         system_text = system_summaries[model_file]
# #         print("oksysttemtest")
# #         scores = scorer.score(system_text, model_text)
# #         print("ok",scores)
# #         results[model_file] = scores
# #         print("ok",scores)
# for i in range(min(len(model_files), len(system_files))):
#     model_file = model_files[i]
#     system_file = system_files[i]
    
#     model_text = model_summaries[model_file]
#     system_text = system_summaries[system_file]
    
#     # Calculate ROUGE scores
#     scores = scorer.score(system_text, model_text)
#     results[f'{model_file} vs {system_file}'] = scores

# # Print the ROUGE scores for each file pair
# for file, score in results.items():
#     print(f"Scores for {file}:")
#     print(score)

import torch

print("ROUGE setup successful!")
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Cuda setup successful!")
else:
    device = torch.device("cpu")
    print("Cpu setup successful!")
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
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer

from pyrouge import Rouge155
from rouge_score import rouge_scorer
from preprocess import DatasetProcessor
from models import BertClassifier



# Assuming other necessary imports (such as DatasetProcessor, model initialization, etc.) are present
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
# def predict_sample(model, sample_text, processor, tokenizer, device, topk=5):
#     model.eval()  # Set model to evaluation mode

#     # Preprocess the input sample text (split into sentences, or you can use a sentence tokenizer if needed)
#     sentences = sample_text.split('\n')  # Simple split by new lines, adjust based on your input format
    
#     # Prepare a list for storing sentence scores
#     predicted_sentences = []
    
#     # Get all comment embeddings (if needed)
#     all_comments_emb = processor.get_all_features_of_comments(sample_text)
    
#     for sent in sentences:
#         if len(sent.strip()) == 0:  # Skip empty sentences
#             continue
        
#         # Tokenize the text with the title (if needed)
#         text = tokenizer.cls_token + sent + tokenizer.sep_token  # Simple format for now
        
#         # Convert to model input features
#         input_ids, attention_masks = processor.convert_example_to_feature(text)
        
#         # Get feature of the best comment (if necessary)
#         comment_feature = processor.get_feature_of_best_comment(sent, all_comments_emb)
        
#         # Forward pass through the model
#         logits = model(input_ids.to(device), attention_masks.to(device), comment_feature.to(device))
        
#         # Apply softmax to get the prediction scores
#         score = F.softmax(logits, dim=1)
        
#         # Store sentence and its score
#         predicted_sentences.append((sent, score[0][1].item()))  # Assuming the score for sentence relevance is at index 1

#     # Sort sentences based on the prediction score in descending order
#     predicted_sentences = sorted(predicted_sentences, key=lambda x: x[1], reverse=True)

#     # Select the top-k sentences based on the score
#     top_sentences = predicted_sentences[:topk]

#     # Extract the sentences to form the summary
#     summary = [sentence[0] for sentence in top_sentences]
    
#     # Return the generated summary
#     return '\n'.join(summary)


# def main():
#     # Example input text (This could be passed as an argument or loaded from a file)
#     sample_text = '''
#     The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price.
#     The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal.
#     Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008.
#     Real estate firm Tishman Speyer had owned the other 10%.
#     The buyer is RFR Holding, a New York real estate company.
#     Officials with Tishman and RFR did not immediately respond to a request for comments.
#     It's unclear when the deal will close.
#     The building sold fairly quickly after being publicly placed on the market only two months ago.
#     The sale was handled by CBRE Group.
#     The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.
#     The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028.
#     Meantime, rents in the building itself are not rising nearly that fast.
#     While the building is an iconic landmark in the New York skyline, it is competing against newer office towers with large floor-to-ceiling windows and all the modern amenities.
#     Still the building is among the best known in the city, even to people who have never been to New York.
#     It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.
#     It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.
#     The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
#     Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.
#     Blackstone Group (BX) bought it for $1.3 billion 2015.
#     The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.
#     Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.
#     Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title.
#     '''

#     # Initialize the model and processor (make sure the model is already loaded and the processor is set up correctly)
#     # Example of initializing the model and tokenizer
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Use your pre-trained model
#     processor = DatasetProcessor(args, tokenizer)  # Make sure args and processor are defined

#     # Assume the model has been trained and is ready to be used for prediction
#     # Initialize your model here
#     model = torch.load("saved_models/bert_classifier_fold_5.pt")  # Adjust with the correct model path
#     nn.ModuleList(model)
#     #model.to(device)  # Move the model to the correct device (e.g., GPU or CPU)

#     # Make prediction
#     summary = predict_sample(model, sample_text, processor, tokenizer, device, topk=5)
    
#     print("Generated Summary:")
#     print(summary)


# if __name__ == "__main__":
#     main()



# def initialize_model(epochs=args.epochs):
#     bert_classifier = BertClassifier(args)
#     bert_classifier.to(device)
#     optimizer = AdamW(bert_classifier.parameters(), lr=args.lr, eps=1e-8)
#     total_steps = args.epochs
#     scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                 num_warmup_steps=50,
#                                                 num_training_steps=total_steps)
#     loss_fn = nn.CrossEntropyLoss()
#     return bert_classifier, optimizer, scheduler, loss_fn
# # Load the model state dictionary from the .pt file
# tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)
# processor = DatasetProcessor(args, tokenizer)
# model_state_dict = torch.load('saved_models/bert_classifier_fold_5.pt')
# bert_classifier = BertClassifier(args)
# bert_classifier.to(device)
# bert_classifier.load_state_dict(model_state_dict)
# bert_classifier.eval()


# # Print the contents of the state dictionary
# for param_name, param_value in model_state_dict.items():
#     print(f"Parameter Name: {param_name}")
#     print(f"Parameter Shape: {param_value.shape}")
    
    
# sample ='''
# The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price.
# The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal.
# Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008.
# Real estate firm Tishman Speyer had owned the other 10%.
# The buyer is RFR Holding, a New York real estate company.
# Officials with Tishman and RFR did not immediately respond to a request for comments.
# It's unclear when the deal will close.
# The building sold fairly quickly after being publicly placed on the market only two months ago.
# The sale was handled by CBRE Group.
# The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.
# The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028.
# Meantime, rents in the building itself are not rising nearly that fast.
# While the building is an iconic landmark in the New York skyline, it is competing against newer office towers with large floor-to-ceiling windows and all the modern amenities.
# Still the building is among the best known in the city, even to people who have never been to New York.
# It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.
# It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.
# The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
# Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.
# Blackstone Group (BX) bought it for $1.3 billion 2015.
# The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.
# Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.
# Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title.
# '''

# with torch.no_grad():
#     prediction = bert_classifier(sample)


import torch
from transformers import BertTokenizer
import torch.nn.functional as F

# Assuming you have the BertClassifier class definition from earlier
# You'll need to create an argparse-like object to pass to the model initialization
class Args:
    def __init__(self):
        self.pretrained_bert = 'bert-base-uncased'
        self.kernel_sizes = '1,2,3'
        self.kernel_num = 100
        self.dropout = 0.5

# Your full text
text = """It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.
It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.
The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.
Blackstone Group (BX) bought it for $1.3 billion 2015.
The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.
Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.
Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title."""

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input text
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

# Load the model
args = Args()
model = BertClassifier(args)
model.load_state_dict(torch.load('saved_models/bert_classifier_fold_5.pt', map_location=torch.device('cpu')))
model.eval()

# Prepare additional features (adjust as needed)
comments_feature = torch.zeros(1, 768)  # Placeholder

# Perform summarization
with torch.no_grad():
    outputs = model(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        comments_feature=comments_feature
    )

# Process the outputs
probabilities = F.softmax(outputs, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

# Print results
print("Model Outputs:", outputs)
print("Probabilities:", probabilities)
print("Predicted Class:", predicted_class)

# If this is an extractive summarization model, you might want to:
# 1. Identify key sentences
# 2. Extract sentences based on the model's predictions
def extract_summary(text, model_outputs, tokenizer, top_k=2):
    # Split text into sentences
    sentences = text.split('.')
    
    # You'll need to implement logic to select top sentences based on model outputs
    # This is a placeholder approach
    selected_sentences = sentences[:top_k]
    
    return '. '.join(selected_sentences) + '.'

# Generate summary
summary = extract_summary(text, outputs, tokenizer)
print("\nGenerated Summary:")
print(summary)