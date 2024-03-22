# from tqdm import tqdm
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms


from data_loader import get_loader
# from data_loader import my_get_loader
from models import VqaModel

def loadPkl(file_path, delimiter='\n'):
	with open(file_path, "rb") as f:
		data = pkl.load(f)
	return data
def getWord2IdxMap(idx2word):
	word2idx = {}
	for idx, word in enumerate(idx2word):
		word2idx[word]=idx
	return word2idx	


def loadTxtFile(file_path, delimiter='\n'):
	""" Function to load text files as numpy arrays
	Inputs: 
	    file_path: path to input file
	    delimter: (optional) delimiter to be used. "\n" by default
	Return Value: An array containing data from the text file, seperated by delimiter
	"""
	with open(file_path, "r") as f:
		arr = f.read()
	return arr.split(delimiter)

def readImage(path):
	return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def showImage(img):
	plt.imshow(img)
	plt.savefig('my_figure.png')  # Save the figure to a file
	#plt.show()
	

def resizeImage(img, target_size=(224,224)):
	return cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)

def normalizeImage(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
	img = img / 255
	img = img.astype("float32")

	r = img[:,:,0]
	g = img[:,:,1]
	b = img[:,:,2]

	x = (r-mean[0])/std[0]
	y = (g-mean[1])/std[1]
	z = (b-mean[2])/std[2]
	normalized_img = np.stack([x,y,z], axis=2)
	return normalized_img

def transformImage(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], target_size=(224,224)):
	resized_img = resizeImage(img, target_size)
	normalized_img = normalizeImage(resized_img, mean, std)
	transposed_img = np.transpose(normalized_img, axes=[2,0,1])
	m,n,p = transposed_img.shape
	transformed_img = transposed_img.reshape(1,m,n,p)
	return torch.from_numpy(transformed_img)


def encodeQuestion(qst_str, qst_word2idx, max_qst_length):
	qst_str = qst_str.strip()
	qst_words = qst_str.split(" ")
	count=0

	my_question = []
	for qst_word in (qst_words):
		my_question.append(qst_word2idx[qst_word])
		count+=1

	while count < max_qst_length:
		my_question.append(0)
		count+=1

	my_question = np.asarray(my_question)
	my_question = my_question.reshape((1,-1))
	qst_tensor = torch.from_numpy(my_question)

	return qst_tensor	

def getPrediction(img, qst_str):
	### Make Predictions ###
	transformed_img = transformImage(img)
	qst_encoded = encodeQuestion(qst_str, qst_word2idx, max_qst_length)
	my_output = model(transformed_img, qst_encoded)
	_, my_pred = torch.max(my_output, 1)
	predicted_answer = ans_idx2ans[my_pred]
	print("Question was: ", qst_str)
	print("Predicted Answer is:", predicted_answer)
	showImage(img)

if __name__ == "__main__":
	### Define global variables ###
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = 'cpu'
	print("device: ", device)

	qst_idx2word = loadTxtFile("./datasets/vocab_questions.txt", delimiter="\n")
	qst_idx2word = np.asarray(qst_idx2word)[:-1]
	ans_idx2ans = loadTxtFile("./datasets/vocab_answers.txt", delimiter="\n")
	ans_idx2ans = np.asarray(ans_idx2ans)[:-1]
	qst_word2idx = getWord2IdxMap(qst_idx2word)
	# qst_word2idx = loadPkl("../data_dump/vqa_binary_processed/mapping_questions_word2idx.pkl")


	qst_vocab_size = len(qst_idx2word)
	ans_vocab_size = len(ans_idx2ans)
	max_qst_length = 30
	ans_unk_idx = 0

	# qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
	# ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
	# ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx


	embed_size = 1024
	word_embed_size = 300
	num_layers = 2
	hidden_size = 512

	### Define and load model ###

	model = VqaModel(
	    embed_size=embed_size,
	    qst_vocab_size=qst_vocab_size,
	    ans_vocab_size=ans_vocab_size,
	    word_embed_size=word_embed_size,
	    num_layers=num_layers,
	    hidden_size=hidden_size).to(device)

	model.load_state_dict(torch.load("./models/model-epoch-03.ckpt", map_location=torch.device(device))['state_dict'])

	
	# print("Printing model architecture:\n", model.eval())

	# ### Read / Get input ###
	img = readImage("./datasets/Images/test2015/COCO_test2015_000000000019.jpg")
	# img = readImage("./datasets/Images/bag.jpeg")
	#qst_str = input("Enter a question: ")
	qst_str = "where is the animal in the image ?"
	getPrediction(img, qst_str)