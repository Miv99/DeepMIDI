import argparse
import pickle
import os

import common as c
import emotions_data

import numpy as np
from music21 import *


def read_midi_data(file):
	'''
	Reads all the notes and chords from the specified file.
	
	Returns a list of elements where each element is a tuple: 
		(offset, [pitches], duration, emotions tuple)
	where times are in quarter beats
	and emotions tuple is all zeros.
	'''
	
	data = []
	emotions = [0] * emotions_data.NUM_EMOTIONS
	
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
		pitches = c.element_to_pitches(element)
		# Ignore elements with stupidly long durations (8 or more beats)
		if len(pitches) > 0 and element.duration.quarterLength < 32:
			data.append((element.offset, pitches, element.duration.quarterLength, emotions))
		
	# Sort data ascending by element start time; elements at the same time are by ascending pitch
	return sorted(data)

def generate_notes(model, time_slices, min_percent_for_activation, desired_emotions, duration, initial_state=[]):
	'''
	desired_emotions - an iterable of the emotions to guide the note generation.
		Order is detailed in emotions_data.py.
	duration - the number of time steps in the generated song
	initial_state - list of tuples: (set[pitches on at current time step], set[pitches just activated at current time step])
	
	Outputs a list of tuples: 
	(set[pitches on at current time step], set[pitches just activated at current time step])
	'''
	
	if len(initial_state) == 0:
		# Pick a random sequence from the input as a starting point for the prediction
		pattern = c.get_one_input_output(time_slices, -1)[0]
	elif len(initial_state) < c.SEQUENCE_LENGTH:
		# Pick a random sequence from the input as a starting point for the prediction
		# then append the initial state
		pattern = np.append(c.get_one_input_output(time_slices, -1)[0], initial_state, axis=0)
		pattern = pattern[-c.SEQUENCE_LENGTH:]
	else:
		pattern = initial_state[-c.SEQUENCE_LENGTH:]
					
	# Adjust pattern so that its emotions data match the input emotions
	for i in range(pattern.shape[0]):
		for k, emotion in enumerate(desired_emotions):
			pattern[i][256 + k] = emotion
	
	prediction_output = []
		
	for note_index in range(duration):
		# prediction_input shape is 1 x SEQUENCE_LENGTH x NUM_FEATURES
		prediction_input = np.reshape(pattern, (1, c.SEQUENCE_LENGTH, c.NUM_FEATURES))
		
		prediction = model.predict(prediction_input, verbose=0)
		
		pitches_on = set()
		pitches_just_activated = set()
		
		next = []
		for i in range(128):
			if prediction[0][i] >= min_percent_for_activation:
				next.append(1)
				pitches_on.add(i)
			else:
				next.append(0)
		for i in range(128, 256):
			if prediction[0][i] >= min_percent_for_activation and next[i - 128] == 1:
				next.append(1)
				pitches_just_activated.add(i - 128)
			else:
				next.append(0)
		next.extend(desired_emotions)
						
		prediction_output.append((pitches_on, pitches_just_activated))
		
		pattern = np.append(pattern, [next], axis=0)
		pattern = pattern[1:]

	return prediction_output

def create_midi(prediction_output, file_name):
	'''
	prediction_output - list of tuples: 
		(set[pitches on at current time step], set[pitches just activated at current time step])
	'''

	offset = 0
	output_notes = []
	
	# Any pitch on at time step t but not at t-1 must set its just-activated at t
	for i in range(1, len(prediction_output)):
		for pitch in prediction_output[i][0]:
			if pitch not in prediction_output[i - 1][0]:
				prediction_output[i][1].add(pitch)

	for i, (pitches_on, pitches_just_activated) in enumerate(prediction_output):
		for pitch in pitches_just_activated:
			if pitch in pitches_on:
				# Calculate note duration
				duration = 1 # in units of 1/TIME_SLICE_STEP quarter beats
				for k in range(i + 1, len(prediction_output)):
					if (pitch not in prediction_output[k][0]) or (pitch in prediction_output[k][1]):
						break
					else:
						duration += 1
									
				new_note = note.Note(pitch, quarterLength=duration/c.TIME_SLICE_STEP)
				new_note.offset = offset
				new_note.storedInstrument = instrument.Piano()
				output_notes.append(new_note)
		
		# Increase offset
		offset += 1/c.TIME_SLICE_STEP

	midi_stream = stream.Stream(output_notes)
	
	# Create output folder if it doesn't exist
	if not os.path.exists('output'):
		os.makedirs('output')

	midi_stream.write('midi', fp='output/' + file_name + '.mid')
	
	print('Created midi file:' + file_name)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Flag for looping the prompt
	parser.add_argument('-l', action='store_true', help='A flag for looping the prompt when done generating')
	# Fixing input initial MIDI
	parser.add_argument('-i', action='store', type=str, help='An optional string specifying the MIDI file used for the predictions\' initial state. If this is not set, the program will ask for a new value every loop. If that new value is not set, a random initial state will be taken from "data/midi_data", which is generated from trainer.py')
	parser.add_argument('--use_start', action='store_false', help='A flag for using only the first part of an input MIDI file. This is true by default.')
	parser.add_argument('--use_end', action='store_true', help='A flag for using only the last part of an input MIDI file. This is false by default.')

	# Fixing output file name
	parser.add_argument('-o', action='store', type=str, help='An optional string specifying the output file name. If this is not set, the program will ask for a new value every loop.')
	# Fixing happiness
	parser.add_argument('-e1', action='store', type=int, help='An optional int in range [1, 10] specifying the desired happiness. If this is not set, the program will ask for a new value every loop.')
	# Fixing excitement
	parser.add_argument('-e2', action='store', type=int, help='An optional int in range [1, 10] specifying the desired excitement. If this is not set, the program will ask for a new value every loop.')
	# Fixing complexity
	parser.add_argument('-c', action='store', type=float, help='An optional float in range [0, 1] specifying the desired complexity. If this is not set, the program will ask for a new value every loop.')
	# Fixing duration
	parser.add_argument('-d', action='store', type=int, help='An optional positive int specifying the desired duration. If this is not set, the program will ask for a new value every loop.')

	args = parser.parse_args()
	

	with open('data/midi_data', 'rb') as filepath:
		data = pickle.load(filepath)
	time_slices = c.get_time_slices(data)
	
	model = c.create_model()
	model.load_weights('weights.hdf5')
	
	while True:
		if args.e1 is None:
			emotion_1 = int(input('Enter desired song happiness (an integer in range [1, 10]): '))/10
		else:
			emotion_1 = args.e1
			
		if args.e2 is None:
			emotion_2 = int(input('Enter desired song excitement (an integer in range [1, 10]): '))/10
		else:
			emotion_2 = args.e2
		
		if args.c is None:
			complexity = float(input('Enter desired song complexity (a float in range [0, 1]): '))
		else:
			complexity = args.c
			
		if args.d is None:
			duration = int(input('Enter desired duration (in time steps): '))
		else:
			duration = args.d
		
		if args.o is None:
			output_file_name = input('Enter output file name: ')
		else:
			output_file_name = args.o
			
		if args.i is None:
			input_file = input('Enter input file name (leave empty to use a random one from "data/midi_data" generated by trainer.py): ')
			
			if input_file == '':
				input_data_as_time_slices = []
			else:
				input_data = read_midi_data(input_file)
				input_data_as_time_slices = c.get_time_slices(input_data)
		else:
			input_file = args.i
			
			input_data = read_midi_data(input_file)
			input_data_as_time_slices = c.get_time_slices(input_data)
			
		if args.use_start:
			input_data_as_time_slices = input_data_as_time_slices[:c.SEQUENCE_LENGTH]
		elif args.use_end:
			input_data_as_time_slices = input_data_as_time_slices[-c.SEQUENCE_LENGTH:]
		
		# Scale complexity to a percentage between 0.1 and 0.3
		min_activation_percent = complexity*(0.3 - 0.1) + 0.1
				
		prediction = generate_notes(model, time_slices, min_activation_percent, (emotion_1, emotion_2), duration, input_data_as_time_slices)
		
		create_midi(prediction, output_file_name)
		
		if not args.l:
			break