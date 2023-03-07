import random #for choosing random responses
import json	#for loading the json file
import pickle #serialization of python objects
import numpy as np # for array operations
import nltk #natural language toolkit

from nltk.stem import WordNetLemmatizer #lemmatize - reduce words to their root	form/stem

from tensorflow import keras #deep learning library
from keras.models import Sequential #sequential model
from keras.layers import Dense, Activation, Dropout #layers dense layer, activation function, dropout layer
from keras.optimizers import SGD #stochastic gradient descent

def prepare():
	#prepare data

	#lematize individual	words
	lemmatizer = WordNetLemmatizer()
	#load intents
	intents = json.loads(open('intents.json').read())

	#word lists
	words	= []
	classes	= []
	documents = []
	#letters to	ignore
	ignore_letters = ['?', '!', '.', ',']

	#iterate intents
	for intent in intents['intents']:
		#iterate patterns

		for pattern in intent['patterns']:
			#split each sentense into words
			w = nltk.word_tokenize(pattern)
			#add words to word list
			words.extend(w)
			#add words to documents list
			documents.append((w, intent['tag']))
			#add to classes list
			if intent['tag'] not in classes:
				classes.append(intent['tag'])

	#lemmatize + lower case each word + remove duplicates + sort
	words =	[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
	words	= sorted(list(set(words)))
	classes = sorted(list(set(classes)))

	#save data to pickle files
	pickle.dump(words, open('words.pkl', 'wb'))
	pickle.dump(classes, open('classes.pkl', 'wb'))

	#create training data
	training = []
	output_empty = [0] * len(classes)

	#bag of words
	for doc in documents:
		bag = []
		pattern_words = doc[0]
		pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
		for w in words:
			bag.append(1) if w in pattern_words else bag.append(0)

	#set output to '0' for each tag and '1' for current tag (for each pattern)
		output_row = list(output_empty)
		output_row[classes.index(doc[1])] = 1

		training.append([bag, output_row])

	#shuffle features and turn into np.array
	random.shuffle(training)
	training = np.array(training)

	#create train and test lists
	train_x = list(training[:, 0])
	train_y = list(training[:, 1])

	#build model
	model = Sequential()
	model.add(Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(len(train_y[0]), activation = 'softmax'))

	#compile model
	sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
	model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

	#fit model
	hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)
	#save model
	model.save('chatbot_model.h5', hist)


