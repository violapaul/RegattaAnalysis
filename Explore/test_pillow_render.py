

# import required classes

import os
import time
from PIL import Image, ImageDraw, ImageFont

# create Image object with the input image

# create font object with the font file and specify
# desired size
font = ImageFont.truetype('Tahoma.ttf', size=45)


# initialise the drawing context with
# the image object as background

start = time.perf_counter()
image = Image.open('/Users/viola/Downloads/background.png').convert('RGBA')
draw = ImageDraw.Draw(image)

# starting position of the message

(x, y) = (50, 50)
message = "Happy Birthday!"
color = 'rgb(255, 255, 0)' # 

# draw the message on the background

draw.text((x, y), message, fill=color, font=font)

(x, y) = (150, 150)
name = 'Vinay'
color = 'rgb(255, 255, 255)' # white color
draw.text((x, y), name, fill=color, font=font)

print(f"Elapsed time {time.perf_counter() - start}")

# save the edited image

image.save('/tmp/greeting_card.png')
os.system("open /tmp/greeting_card.png")
