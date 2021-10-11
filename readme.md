# TFRecord Writer
## introduce
TFRecord Writer is module helps to use tfrecord easier. 
Write make_example and write_tfrecord_file, then convert your file to
tfrecord file that easily upload I/O FREE tfrecord file which can be used
in Google Colaboratory and Jetbrains DataGrip, and even local machine.

## installation
```bash
!pip install -U git+https://github.com/Klassikcat/TFRecord-converter
```

## Dependency
- Tensorflow == 2.6.0 or higher
- tqdm == 4.62.0 or higher
- pandas == 1.2.5
- numpy == 1.19.2

## Stack
<img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/> <img alt="TensorFlow" src ="https://img.shields.io/badge/TensorFlow-FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white"/> 


## Structure
```angular2html
tfrecord_converter
 ┖ config.py
 ┖ tfrecord_converter.py
```

## How to use
first, you need to write tfrecord example rule method. 

for example, if you want to write tfrecord file that consists with audio file, file_path and
text target data, you need to write feature dictionary and tf.train.Example methods like
below.

```python
import tensorflow as tf
from TFRecord_Converter.config import _float_feature, _bytes_feature, _int64_feature 

def make_example(audio, audio_path, text_target) -> tf.train.Example:
    feature = {'audio/encoded': _float_feature(audio),
               'class/file_path': _bytes_feature(audio_path),
               'class/text': _int64_feature(text_target)
               }
    return tf.train.Example(features=tf.train.Features(feature=feature))
```
choosing feature type among `float feature`, `bytes feature`, and `int64 feature` is all up to you. but
i personlally recommand to use `float` or `integer` feature for audio/image, and `byte` or `int` feature
for label because it is easier to use.

then, define tfrecord_writer which write your file and labels into your tfrecord shards. 

```python
def tfrecord_writer(df, index, example_rule):
    audio = open(df.path.iloc[index], 'rb').numpy().flatten().tolist() # open audio file and encode to floating number 
    file_path = df.file_path.iloc[index].encode(encoding='utf-8') # encode file_path to utf-8 bytes
    text = df.text.iloc[index].encode(encoding='utf-8').encode(encoding='utf-8')
    text = int.from_bytes(bytes=text, byteorder='big')
    example = example_rule(audio=audio, audio_path=file_path, text_target=text)
    return example
```

lastly, assign dataframe, file path, tfrecord_writer, example_rule, SEED, n_shards and tfrecord_rules and
run `convert()`. In the below cell, 10 tfrecord shards will be generated in `output/dir/here/`. 

```python
from tfrecord_converter.tfrecord_converter import Converter

converter = Converter(tfrecord_writer=tfrecord_writer, example_rule=make_example,
                      origin_path='data/path/here/', output_dir='output/dir/here/',
                      df=your_dataframe_here, n_shards=10, SEED=42)
converter.convert()
```