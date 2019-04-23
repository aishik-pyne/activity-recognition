from keras import Model
from keras.layers import MaxPooling2D, Flatten
from keras.applications import vgg16


def vgg16_flattened(image_shape=(120, 160, 3)):
  """
  Returns a pretrained model as the base cnn
  """
  model = vgg16.VGG16(include_top=False, weights='imagenet',
                      input_shape=image_shape, pooling=None)
  x = model.output
  # x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)
  x = Flatten()(x)
  model = Model(inputs=model.input, outputs=x)
  for layers in model.layers:
    layers.trainable = False
  return model
