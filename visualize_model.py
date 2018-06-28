from keras.utils.vis_utils import plot_model
from keras.models import load_model

model = load_model('models/model.h5')

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)