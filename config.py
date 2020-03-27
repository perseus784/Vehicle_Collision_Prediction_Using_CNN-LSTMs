import os
stride_size=1
height = 50
width = 50
color_channels = 1
epochs = 10
image_path = os.path.join("datasets","formatted")

vocabulary_size = 355
embedding_layer_size= 48
#slots =
number_of_intents=4
number_of_seqs=200001
max_len = 20
decoder_max_len=max_len
number_of_epochs = 3
number_of_hiddenunits = 80
batch_size = 32
time = 10
