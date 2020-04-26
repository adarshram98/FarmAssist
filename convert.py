# Import the Gtts module for text  
# to speech conversion 
from gtts import gTTS 
  
# import Os module to start the audio file
import os 

fh = open("test.txt", "r")
myText = fh.read().replace("\n", " ")

# Language we want to use 
language = 'en'

output = gTTS(text=myText, lang=language, slow=False)

output.save("output.mp3")
fh.close()

# Play the converted file 
os.system("start output.mp3")

