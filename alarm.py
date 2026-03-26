from gtts import gTTS

tts = gTTS("Animal detected", lang="en")
tts.save("alert.mp3")

print("Audio created successfully")