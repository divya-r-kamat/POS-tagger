# Part of Speech Tagger
This notebook contains code for neural network that can tag POS in an English sentence. There are many POS tagsets available, here universal tagset has been used. The model converts the sentence to POS tags. Tags used are:

**ADJ - Adjective<br>
ADP - Adposition
ADV - Adverb<br>
PRT -	Particle<br> 
PRON - Pronoun<br>
.	   - Punctuation marks<br>
X	- Other	<br>
VERB - Verb<br>
CONJ	- Conjunction<br>
DET - Determiner / Article	
NOUN	- Noun	<br>
NUM - Numeral<br>**

## Model Architecture
It uses a bidirectional LSTM model. The model achieved a validation accuracy of 96% on validation data.
<p align=center>
  <img src="media/model_plot.png" width=500 height=500>
</p>

The model can be improved even further. This model was trained only for 10 epochs. Further accuracy can be improved by increasing the number of hidden state units, stacking up more layers, using pretrained word embeddings etc. 

This code is inspired from [here](https://www.coursera.org/learn/language-processing).
