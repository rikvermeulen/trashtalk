import numpy as np
import pyaudio as pa
import os, time

#file imports 
import config
import model_setup
from frame_asr import FrameASR

asr = FrameASR(model_definition = {
                   'sample_rate': config.SAMPLE_RATE,
                   'AudioToMelSpectrogramPreprocessor': model_setup.cfg.preprocessor,
                   'JasperEncoder': model_setup.cfg.encoder,
                   'labels': model_setup.cfg.decoder.vocabulary
               },
               frame_len=config.FRAME_LEN, frame_overlap=2, 
               offset=4)

asr.reset()

p = pa.PyAudio()
print('Available audio input devices:')
input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        input_devices.append(i)
        print(i, dev.get('name'))

if len(input_devices):
    dev_idx = -2
    while dev_idx not in input_devices:
        print('Please type input device ID:')
        dev_idx = int(input())

    empty_counter = 0

    def callback(in_data, frame_count, time_info, status):
        global empty_counter
        signal = np.frombuffer(in_data, dtype=np.int16)
        text = asr.transcribe(signal)
        if len(text):
            print(text,end='')
            empty_counter = asr.offset
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                print(' ',end='')
        return (in_data, pa.paContinue)

    stream = p.open(format=pa.paInt16,
                    channels=config.CHANNELS,
                    rate=config.SAMPLE_RATE,
                    input=True,
                    input_device_index=dev_idx,
                    stream_callback=callback,
                    frames_per_buffer=config.CHUNK_SIZE)

    print('Listening...')

    stream.start_stream()
    
    # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
    try:
        while stream.is_active():
            time.sleep(0.1)
    finally:        
        stream.stop_stream()
        stream.close()
        p.terminate()

        print()
        print("PyAudio stopped")
    
else:
    print('ERROR: No audio input device found.')
