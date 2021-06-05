import glob
import pickle
import numpy as np
import math
import keras
import emotions_data
from music21 import *

# The smallest possible generated note duration
# Each time slice will be 1/TIME_SLICE_STEP quarter beats
TIME_SLICE_STEP = 4

# Model batch size
BATCH_SIZE = 128
# Number of previous time slices used to predict the next time slice
SEQUENCE_LENGTH = TIME_SLICE_STEP * 4 * 8	# 8 measures, assuming 4/4 time

# (multi-hot encoding of element pitches) + (pitch activation bools, one for each pitch) + (emotions)
NUM_FEATURES = (128) + (128) + (emotions_data.NUM_EMOTIONS)
# (multi-hot encoding of element pitches) + (pitch activation bools, one for each pitch)
NUM_OUTPUT_DIMENSIONS = (128) + (128)

		
def get_time_slices(data):	
	'''
	data - The pickled combined MIDI data
	
	Returns the data as time slices: a 2D array with shape
	(number of time steps, number of features)
	'''

	# Scale all the time values in data so that now time units are in 1/TIME_SLICE_STEP quarter beats
	scaled_time_data = [(element[0] * TIME_SLICE_STEP, element[1], element[2] * TIME_SLICE_STEP, element[3]) for element in data]
	
	# Max time before the last note being played ends
	max_time = max([x[0] + x[2] for x in scaled_time_data[-100:]])
		
	# Array of shape (# time steps, features)
	time_slices = np.zeros((math.ceil(max_time), NUM_FEATURES))
		
	for element in scaled_time_data:
		time_start = element[0]
		pitches = element[1]
		duration = element[2]
		emotions = element[3]
		just_activated = 0
		
		if abs(round(time_start) - time_start) < 1e-3:
			just_activated = 1
		
		set_just_activated = False
		for t in range(round(time_start), int(time_start + duration), 1):
			for pitch in pitches:
				time_slices[t][pitch] = 1
				if not set_just_activated:
					time_slices[t][128 + pitch] = just_activated
					set_just_activated = True
			for i in range(emotions_data.NUM_EMOTIONS):
				time_slices[t][256 + i] = emotions[i]
	
	# If any pitches are activated in one time step but not the previous, set its just_activated to 1
	for t in range(1, len(time_slices)):
		for pitch in range(128):
			if time_slices[t][pitch] > 0 and time_slices[t - 1][pitch] == 0:
				time_slices[t][128 + pitch] = 1
				
	return time_slices
	
def element_to_pitches(element):
	if isinstance(element, note.Note):
		return [int(element.pitch.ps)]
	elif isinstance(element, chord.Chord):
		return sorted([int(pitch.ps) for pitch in element.pitches])
	else:
		return []
		
def get_one_input_output(time_slices, start):
	'''
	start - start index for time_slices; -1 for random
	
	Returns a tuple: (model input, model output)
	'''
	if start == -1:
		start = np.random.randint(0, len(time_slices) - 1)

	i = start
	N = len(time_slices) - SEQUENCE_LENGTH

	model_input = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES))
	output = np.zeros(NUM_OUTPUT_DIMENSIONS)
	
	for k in range(SEQUENCE_LENGTH):
		# Set input
		model_input[k] = time_slices[i + k]
		
	# Output is just the pitches and their just_activateds
	output = time_slices[i + SEQUENCE_LENGTH][:256]
		
	return (model_input, output)

def create_model():
	input_layer = keras.Input(
		shape=(SEQUENCE_LENGTH, 256 + emotions_data.NUM_EMOTIONS),
		name='input'
	)
	
	lstm_1 = keras.layers.LSTM(256, return_sequences=True)(input_layer)
	lstm_2 = keras.layers.LSTM(256)(lstm_1)
	dense_1 = keras.layers.Dense(256)(lstm_2)
	output = keras.layers.Activation('sigmoid', name='output')(dense_1)
	
	model = keras.Model(
		inputs=[input_layer],
		outputs=[output]
	)
	
	model.compile(
		loss=['binary_crossentropy'],
		loss_weights=[100.0],
		optimizer='rmsprop'
	)
			
	return model