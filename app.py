import soundfile as sf
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
import torch
# Download and load the pretrained fastpitch model
spec_generator = SpectrogramGenerator.from_pretrained(model_name="tts_en_fastpitch").cuda()
# Download and load the pretrained hifigan model
vocoder = Vocoder.from_pretrained(model_name="tts_hifigan").cuda()

# All spectrogram generators start by parsing raw strings to a tokenized version of the string
parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
# They then take the tokenized string and produce a spectrogram
spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
# Finally, a vocoder converts the spectrogram to audio
audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# Save the audio to disk in a file called speech.wav
# Note vocoder return a batch of audio. In this example, we just take the first and only sample.
sf.write("speech.wav", audio.to('cpu').detach().numpy()[0], 22050)
