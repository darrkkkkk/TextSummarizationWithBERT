import os
import argparse
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from preprocess import DatasetProcessor
from models import BertClassifier
from rouge_score import rouge_scorer
from tqdm import tqdm

def load_model(model_path, args, device):
    """
    Loads the trained BertClassifier model from the specified path.
    
    Args:
        model_path (str): Path to the saved model weights.
        args (Namespace): Parsed command-line arguments.
        device (torch.device): Device to load the model onto.
    
    Returns:
        nn.Module: Loaded BertClassifier model.
    """
    model = BertClassifier(args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_summary(text, model, processor, tokenizer, device, topk=4):
    """
    Generates a summary for the given text using the trained model.
    
    Args:
        text (str): The input text to summarize.
        model (nn.Module): The trained BertClassifier model.
        processor (DatasetProcessor): Data processor instance.
        tokenizer (BertTokenizer): Tokenizer used in preprocessing.
        device (torch.device): Device to perform computations on.
        topk (int, optional): Number of top sentences to select for summary. Defaults to 4.
    
    Returns:
        str: The generated summary.
    """
    # Split the text into sentences
    sentences = split_into_sentences(text)

    predicted_sentences = []
    
    for sent in sentences:
        title = ""  # Assuming no title for single text input
        content = sent
        combined_text = f"{tokenizer.cls_token} {title} {tokenizer.sep_token} {content} {tokenizer.sep_token}"
        
        # Convert text to features
        input_ids, attention_masks = processor.convert_example_to_feature(combined_text)
        
        # Since there are no comments associated with arbitrary text, provide a dummy comments_feature
        # The size of comments_feature should match what the model expects. Adjust the size accordingly.
        # For example, if comments_feature is a tensor of size (1, 768):
        comment_feature = torch.zeros((1, 768)).to(device)  # Adjust the size if different
        
        # Move tensors to device
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        comment_feature = comment_feature.to(device)
        
        # Get model logits
        with torch.no_grad():
            logits = model(input_ids, attention_masks, comment_feature)
            scores = F.softmax(logits, dim=1)
            score = scores[0][1].item()  # Assuming binary classification with index 1 as positive class
        
        predicted_sentences.append((content.lower(), score))
    
    # Sort sentences based on score in descending order
    predicted_sentences = sorted(predicted_sentences, key=lambda x: x[1], reverse=True)
    
    # Select top-k sentences
    selected_sentences = [sentence[0] for sentence in predicted_sentences[:topk]]
    
    # Combine selected sentences into a summary
    summary = ' '.join(selected_sentences)
    
    return summary

def split_into_sentences(text):
    """
    Splits the input text into sentences.
    This is a simple implementation. For more robust sentence splitting, consider using NLTK or SpaCy.
    
    Args:
        text (str): The text to split.
    
    Returns:
        list: List of sentences.
    """
    import re
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    return sentences

def main():
    parser = argparse.ArgumentParser(description='Summarize Text Using Trained BertClassifier')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (e.g., saved_models/bert_classifier_fold_5.pt)')
    parser.add_argument('--text', type=str, required=False, help='Input text to summarize. If not provided, the script will prompt for input.')
    parser.add_argument('--topk', type=int, default=4, help='Number of top sentences to include in the summary')
    parser.add_argument('--save_summary', type=str, default=None, help='Path to save the generated summary. If not provided, the summary will be printed.')
    # Add other arguments as needed, matching those required by DatasetProcessor
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--kernel_num', type=int, default=100, help='Number of kernels per size')
    parser.add_argument('--kernel_sizes', type=str, default='1,2,3', help='Comma-separated kernel sizes')
    parser.add_argument('--pretrained_bert', type=str, default='bert-base-uncased', help='Pre-trained BERT model name')
    parser.add_argument('--pretrained_sentence_bert', type=str, default='bert-base-nli-mean-tokens', help='Pre-trained Sentence-BERT model name')
    parser.add_argument('--rouge_path', type=str, default='/path/to/ROUGE-1.5.5/', help='Path to ROUGE-1.5.5 directory')  # Update if needed
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_sequence_length', type=int, default=128, help='Maximum sequence length for tokenizer')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for K-Fold Cross-Validation')
    parser.add_argument('--train_path', type=str, default='./data/USAToday-CNN.json', help='Path to training data')
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)  # Ensure consistency with training
    
    # Initialize DatasetProcessor
    processor = DatasetProcessor(args, tokenizer)
    
    # Load the trained model
    model = load_model(args.model_path, args, device)
    print(f"Loaded model from {args.model_path}")
    
    # Get input text
    if args.text:
        input_text = args.text
    else:
        print("Please enter the text to summarize:")
        input_text = input(">> ")
    
    # Generate summary
    summary = generate_summary(input_text, model, processor, tokenizer, device, topk=args.topk)
    
    # Output the summary
    if args.save_summary:
        try:
            with open(args.save_summary, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Summary saved to {args.save_summary}")
        except Exception as e:
            print(f"Error saving summary to {args.save_summary}: {e}")
    else:
        print("\n=== Generated Summary ===")
        print(summary)
        print("=========================")

if __name__ == "__main__":
    main()
