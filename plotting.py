import matplotlib.pyplot as plt
import json
import os


inception_json = os.path.join('files','inception','training_logs.json')
vgg_json = os.path.join('files','vgg','training_logs.json')

with open(inception_json,'r') as f:
    inception_data = json.load(f)

with open(vgg_json,'r') as f:
    vgg_data = json.load(f)

plt.plot(inception_data['val_loss'][:40], label='Inception module')
plt.plot(vgg_data['val_loss'], label='VGG' )

plt.legend(loc="upper left")

plt.ylabel('Validation Loss')
plt.show()
