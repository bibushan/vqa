import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from utils import text_helper
from utils.resize_images import resize_image
from utils.make_vacabs_for_questions_answers import make_vocab_questions
from utils.make_vacabs_for_questions_answers import make_vocab_answers
from utils.build_vqa_inputs import vqa_processing
from models import VqaModel
from data_loader import get_loader




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

def main(args):
    # resize the input image
    # Path to the input image
    image_path =  args.image_path ##r"E:\Research\vqa-ssm\datasets\Images\train2014\COCO_train2014_000000000025.jpg"

    # Desired size after resizing
    image_size = (224, 224)

    print("Image path:", image_path)

    # Open the image
    image = Image.open(image_path)

    # Resize the image
    resized_image = resize_image(image, image_size)

    # Save or use the resized image as needed
    resized_image.save("resized_image.jpg")


    # Make vocabularies for questions and answers
#    make_vocab_questions(args.questions_dir)
 #   make_vocab_answers(args.annotations_dir, args.n_answers)

    # Build VQA inputs
#    vqa_processing(args.resized_images_dir, args.annotations_file, args.questions_file, args.vocab_answers_file, 'test-dev2015')


    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx

    # Load the trained model
    model = VqaModel(embed_size=args.embed_size,
                     qst_vocab_size=qst_vocab_size,
                     #qst_vocab_size=text_helper.VocabDict(args.input_dir+'/vocab_questions.txt'),

                     ans_vocab_size=ans_vocab_size,
                     #ans_vocab_size=text_helper.VocabDict(args.input_dir+'/vocab_answers.txt'),
                     word_embed_size=args.word_embed_size,
                     num_layers=args.num_layers,
                     hidden_size=args.hidden_size)
    
    print("Pre load_state_dict")
    
    
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Model has been loaded")  
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process image and question
    image = load_image(args.image_path, transform)
    question = text_helper.tokenize(args.question)


    # Get the answer
    with torch.no_grad():
        output = model(image, question)
        _, answer_idx = torch.max(output, dim=1)
        answer = text_helper.idx_to_answer(answer_idx.item())

    print("Predicted Answer:", answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--vocab_path', type=str, default="./datasets/vocab_questions.txt", help='Path to vocabulary file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--question', type=str, required=True, help='Question about the image')
    parser.add_argument('--embed_size', type=int, default=1024, help='Embedding size')
    parser.add_argument('--qst_vocab_size', type=int, default=1000, help='Question vocabulary size') ### self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
    parser.add_argument('--ans_vocab_size', type=int, default=1000, help='Answer vocabulary size')   ### self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
    parser.add_argument('--word_embed_size', type=int, default=300, help='Word embedding size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of LSTM')
    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')
    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='maximum number of answers.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of processes working on cpu.')
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