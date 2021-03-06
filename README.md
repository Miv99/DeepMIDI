# DeepMIDI
 
This project is a deep learning music generator that is able to be guided by high-level music properties. The user can specify desired happiness, excitement, complexity, and duration of a generated MIDI. The user is also able to specify an input MIDI file such that the generated MIDI is a continuance of either the first or last segment of the input MIDI.

# Installation

All requirements are in requirements.txt. Run
```
pip install -r requirements.txt
```
to install them.

This program was made with Python 3.9.1, but any Python version 3.6+ should work.

# Technical Overview

The model is 2 layers of LSTMs, each with 256 hidden units, followed by a dense layer with 256 units and then a sigmoid activation layer. It takes in as input an array of shape (SEQUENCE_LENGTH, NUM_FEATURES) and outputs a 1D array of length 256. The model splits MIDI data into time slices (the granularity of which is defined in common.py's TIME_SLICE_STEP; by default, each time slice is a 16th beat). A total of SEQUENCE_LENGTH time slices are used to predict what the notes in the next time slice will be.

Each 1D array of length NUM_FEATURES in the input is the features for a single time slice. In a MIDI file, pitches can range from 0 to 127. First, there are 128 features, one for each pitch, with each one indicating whether the corresponding pitch is currently played in the current time slice. Since a differentation must be made between whether a pitch is currently playing or was just turned on (else there would be no difference between 1 long note and 2 short notes of the same pitch), there are an additional 128 features indicating whether the corresponding pitch was just turned on. Then there are 2 more features for the high-level emotions, happiness and excitement, associated with the current song the model is being trained on.

The output is the same as the input, but without the high-level emotions. Because of the sigmoid activation, each output dimension is the probability of the corresponding pitch being on or just turned on. When generating music, the complexity specififed by the user determines the minimum threshold required for a pitch to be part of the output MIDI.

So then training involves a "sliding window" of length SEQUENCE_LENGTH moving across the time slices of the MIDI files to be used in training, with each window trying to predict the next time slice.

Generating music is done the exact same way, but with each output time slice feeding back as the last time slice of the next input. The emotions part of each time slice are kept constant as the values inputted by the user (the desired generated song emotions), and serve as a sort of reminder for how each output time slice should be like.

## Training

The model comes pre-trained on the MIDIs in the midis folder. If you don't want to train it on your own MIDIs, you can skip to the next section.

```
python trainer.py -h
```
will show a help page for the trainer.

#### Pre-processing

The model requires the MIDI files that will be used to train it to first be processed into a certain format. Running
```
python trainer.py -r
```
will process all .mid files in the "midis" folder into a single file, stored in "data/midi_data". All MIDI file names in this folder must also be present in emotions_data.py's dictionary, which maps file name (without the ".mid") to the high-level emotions associated with that MIDI. These high-level emotions must be manually labeled by you and are necessary for training the model.

In order to reduce overfitting during training and increase song coherence during generation, MIDI files are transposed to C major scale during preprocessing.

#### Training

The model will use only "data/midi_data" from the pre-processing section as training data. Every training epoch, a new model weights file will be stored in the "weights" folder. Running
```
python trainer.py
```
with no flags will train the model from scratch. Running with the -c flag
```
python trainer.py -c
```
will continue the training using "weights.hdf5" in the root folder. 

Note that this "weights.hdf5" file is not created automatically; a weights file must be copied from the "weights" folder to the root folder and renamed to "weights.hdf5" if you are continuing training. When you are done training, repeat the copying and renaming process, as the model uses "weights.hdf5" to generate music.

# Generating music

After you have your "weights.hdf5" file in the root folder, running 
```
python predictor.py
```
will load the model and prompt you for how you want your generated music.

Here is a sample interaction:
```
Enter desired song happiness (an integer in range [1, 10]): 4
Enter desired song excitement (an integer in range [1, 10]): 9
Enter desired song complexity (a float in range [0, 1]): 1
Enter desired duration (in time steps): 500
Enter output file name: asd
Enter input file path (leave empty to use a random initial state from "data/midi_data"):
```
Happiness and excitement guide the high-level properties of the generated song and should be integers from 1 to 10. A happiness value of 1 means sad, 5 means neutral, and 10 means happy. For excitement, 1 means calm, 5 means neutral, and 10 means exciting.

Complexity determines the minimum threshold for whether a note should be on at some given time slice (see the technical overview section for more info). Basically, the higher this value is, the higher the note density.

Duration is in number of time slices, which by default is a 16th beat. So a duration of 160 will generate 10 beats.

Output file name is for the file name of the generated MIDI file. Outputs are stored in the "output" folder.

Input file path is the only optional prompt and can be used to initialize the model's state on a specific MIDI file such that the output will try to be a continuance of the input MIDI. Whether the output MIDI continues off the start or the end of the input MIDI is determined by the --use_start and --use_end flags.


All of these prompts can also be specified as command-line options. Check
```
python predictor.py -h
```
for more information.

# Notes and Possibe Improvements

- Training takes a reaaallly long time for ok-sounding results. The pre-trained weights in this repo took ~10 hours on Google Colab's GPUs.
- The model has no recurrent dropout in the LSTMs, but only because having it be non-zero made training take a really long time. However, I didn't notice much overfitting despite this.
- I don't think the emotions axes I chose (sadness/happiness and calmness/excitement) are good indicators of how most people want control over their generated music, but this can always easily be changed, removed entirely, or extended to use different high-level controls.
- Personally, I also don't think the desired happiness/excitement values are actually reflected well in song outputs from this repo's pretrained model. Maybe it needs to be trained more?
- Different parts of songs have different emotions. For my own sanity when labeling training data, the model considers the song as a whole to have the same constant emotions throughout.
