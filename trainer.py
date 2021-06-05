import common as c
import emotions_data

import math
import argparse
import pickle
import glob

import keras
import numpy as np
from music21 import *


def read_midi_data():
	'''
	Reads all the notes and chords from the midi files in the ./midis directory
	
	Returns a list of elements where each element is a tuple: 
		(offset, [pitches], duration, emotions tuple)
	where times are in quarter beats
	and emotions tuple is described in emotions_data.py
	'''
	
	data = []

	# In quarter beats
	cur_song_start_time = 0
		
	for file in glob.glob("midis/*.mid"):
		print("Parsing %s" % file)
		
		# Normalize emotions
		emotions = [x/10 for x in emotions_data.EMOTIONS[file[6:-4]]]
		
		stream = converter.parse(file)
		# Transpose to C major
		k = stream.analyze('key')
		i = interval.Interval(k.tonic, pitch.Pitch('C'))
		stream = stream.transpose(i)
		
		try:
			elements_to_parse = instrument.partitionByInstrument(stream).parts[0].recurse()
		except:
			elements_to_parse = stream.flat.notesAndRests
			
		for element in elements_to_parse:
			pitches = common.element_to_pitches(element)
			# Ignore elements with stupidly long durations (8 or more beats)
			if len(pitches) > 0 and element.duration.quarterLength < 32:
				data.append((cur_song_start_time + element.offset, pitches, element.duration.quarterLength, emotions))
					
		# New song; add some delay
		cur_song_start_time = data[-1][0] + 4
		
		
	# Sort data ascending by element start time; elements at the same time are by ascending pitch
	return sorted(data)
	
def read_and_pickle_midi_data():
	with open('data/midi_data', 'wb') as filepath:
		pickle.dump(read_midi_data(), filepath)

def train_generator(time_slices):
	i = 0
	N = len(time_slices) - c.SEQUENCE_LENGTH

	while True:
		model_input = np.zeros((c.BATCH_SIZE, c.SEQUENCE_LENGTH, c.NUM_FEATURES))
		output = np.zeros((c.BATCH_SIZE, c.NUM_OUTPUT_DIMENSIONS))
		
		for b in range(c.BATCH_SIZE):
			for k in range(c.SEQUENCE_LENGTH):
				# Set input
				model_input[b][k] = time_slices[i + k]
				
			# Output is just the pitches and their just_activateds
			output[b] = time_slices[i + c.SEQUENCE_LENGTH][:256]
			i = (i + 1) % N
		
		yield ({'input': model_input}, {'output': output})

def train(model, time_slices):
	'''
	Trains the model.
	'''

	filepath = 'weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'
	checkpoint = keras.callbacks.ModelCheckpoint(
		filepath,
		monitor='loss',
		verbose=0,
		save_best_only=False,
		mode='min'
	)
	callbacks_list = [checkpoint]

	model.fit(
		x=train_generator(time_slices),
		epochs=2000, 
		batch_size=c.BATCH_SIZE, 
		callbacks=callbacks_list,
		steps_per_epoch=math.ceil(len(time_slices)/c.BATCH_SIZE)
	)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Flag for reading and processing midi files
	parser.add_argument('-r', action='store_true', help='A flag for doing preprocessing on all MIDI files in the "midi" folder')
	# Flag for continuing training
	parser.add_argument('-c', action='store_true', help='A flag for continuing training off of "weights.hdf5"')
	
	args = parser.parse_args()
	
	if args.r:
		read_and_pickle_midi_data()
	else:
		with open('data/midi_data', 'rb') as file:
			data = pickle.load(file)			
		time_slices = c.get_time_slices(data)
		
		model = c.create_model()
		
		if args.c:
			model.load_weights('weights.hdf5')
		
		train(model, time_slices)