import whisper
import time
import warnings
start = time.time()
warnings.filterwarnings('ignore',category=UserWarning)
decode_options={'language':'en'}
model = whisper.load_model(name='tiny.en',device='cuda')
transcription = model.transcribe(audio='D:/Users/mas/Downloads/audio.mp3',**decode_options)
print(transcription['text'])
end = time.time()

print("execution time: ",(end-start)*10**3," ms")