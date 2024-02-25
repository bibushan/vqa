import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from models import VqaModel
from data_loader import VQAImageProcessor
from utils import text_helper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

def main(args):
    # Load the trained model
    model = VqaModel(embed_size=args.embed_size,
                     qst_vocab_size=args.qst_vocab_size,
                     ans_vocab_size=args.ans_vocab_size,
                     word_embed_size=args.word_embed_size,
                     num_layers=args.num_layers,
                     hidden_size=args.hidden_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load vocabulary and other necessary data
    processor = VQAImageProcessor(args.vocab_path)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process image and question
    image = load_image(args.image_path, transform)
    question = processor.process_question(args.question)

    # Get the answer
    with torch.no_grad():
        output = model(image, question)
        _, answer_idx = torch.max(output, dim=1)
        answer = processor.idx_to_answer(answer_idx.item())

    print("Predicted Answer:", answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--question', type=str, required=True, help='Question about the image')
    parser.add_argument('--embed_size', type=int, default=1024, help='Embedding size')
    parser.add_argument('--qst_vocab_size', type=int, default=1000, help='Question vocabulary size') ### self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
    parser.add_argument('--ans_vocab_size', type=int, default=1000, help='Answer vocabulary size')   ### self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
    parser.add_argument('--word_embed_size', type=int, default=300, help='Word embedding size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of LSTM')
    args = parser.parse_args()

    main(args)


"""
import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from models import VqaModel
from data_loader import VQAImageProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

def main(args):
    # Load the trained model
    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=args.qst_vocab_size,
        ans_vocab_size=args.ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load vocabulary and other necessary data
    processor = VQAImageProcessor(
        os.path.join(args.input_dir, 'vocab.json')
    )
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Process image and question
    image = load_image(args.image_path, transform)
    question = processor.process_question(args.question)

    # Get the answer
    with torch.no_grad():
        output = model(image, question)
        _, answer_idx = torch.max(output, dim=1)
        answer = processor.idx_to_answer(answer_idx.item())

    print("Predicted Answer:", answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='Input directory for visual question answering.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--question', type=str, required=True,
                        help='Question about the image')
    parser.add_argument('--embed_size', type=int, default=1024,
                        help='Embedding size')
    parser.add_argument('--qst_vocab_size', type=int, default=1000,
                        help='Question vocabulary size')
    parser.add_argument('--ans_vocab_size', type=int, default=1000,
                        help='Answer vocabulary size')
    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='Word embedding size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden size of LSTM')
    
    args = parser.parse_args()

    main(args)
"""