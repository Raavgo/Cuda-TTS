import pandas as pd
import re
import json
import nltk
import os
import numpy as np
from nltk.tokenize import TweetTokenizer

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.config.shared_configs import BaseAudioConfig

nltk.download('punkt')


def formatter(root_path, manifest_file, **kwargs):    
    items = []
    speaker_name = "Gummibär"

    print(manifest_file)
    print(root_path)

    path = os.path.join(root_path, manifest_file)

    with open(path) as f:
      data = json.load(f)

      for line in data:
        data_line = data[line]
        clean_text = data_line['clean']

        #Lower text for processing
        lower_text = clean_text.lower()

        #Remove punctuations
        lower_text = re.sub("[^-9A-Za-z ]", "" , lower_text)

        #Tokenisation, would be good for text NLP, but for speech it won't work?
        #tokenizer = TweetTokenizer()
        #tokenized_text = tokenizer.tokenize(lower_text)

        audio_file = os.path.join(os.path.join(root_path, "wavs/"), line)
        items.append({"text":lower_text, "audio_file":audio_file, "speaker_name":speaker_name})
      
      return items

output_path = os.path.dirname('/model/')
#First step: Formatting our text.
# - bring everything to lower case
# - remove punctuation
# - No need for: removing stop words, stemming, lemmatisierung, tokenization, one hot encoding,...

#dataset_config = formatter()

path = '../input/metadata-mls/'

dataset_config = BaseDatasetConfig(
    name="dataset", meta_file_train='metadata_mls.json', path=path, ignored_speakers=None)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=23.0,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
    resample=True,
)

config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=3,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    audio=audio_config,
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, test_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
   TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=test_samples
)

trainer.fit()
