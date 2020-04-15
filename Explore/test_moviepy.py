
import os

# Import everything needed to edit video clips
from moviepy.editor import *

from moviepy.config import change_settings, get_setting

print(get_setting("FFMPEG_BINARY"))


video_path = "/Volumes/SSD/Processed/"
file_path = os.path.join(video_path, "output9916.mp4")

video_path = "/Volumes/SSD/VirbEdit/Movies"
file_path = os.path.join(video_path, "VIRB_V0120019.MP4")

# Load myHolidays.mp4 and select the subclip 00:00:50 - 00:00:60
clip = VideoFileClip(file_path).subclip(50,60)

# Reduce the audio volume (volume x 0.8)
clip = clip.volumex(0.8)

# Generate a text clip. You can customize the font, color, etc.
txt_clip = TextClip("My Holidays 2013",fontsize=70,color='white')

# Say that you want it to appear 10s at the center of the screen
txt_clip = txt_clip.set_pos('center').set_duration(10)

# Overlay the text clip on the first video clip
video = CompositeVideoClip([clip, txt_clip])

# Write the result to a file (many options available !)
video.write_videofile("/tmp/foo.mp4")
