from oscilloscope import Oscilloscope

# example formats
supported_image_format = ["png", "jpg", "jpeg"]

image_flag = False

video_file = input('Enter the path of the media file: ')

if video_file.split(".")[-1] in supported_image_format:
    image_flag = True

osc = Oscilloscope(video_file, is_image=image_flag)

osc.convert()

# https://dood.al/oscilloscope/
