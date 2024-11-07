import asyncio
import math

import cv2
import numpy as np
from scipy.io.wavfile import write, read

from tqdm import tqdm

from scipy.spatial import KDTree


def sort_algorithm(points):
    async def distance_calculation(stack, visited):
        """
            use this to find best point to go to next based on distance away from last point in path,
            have to make sure current point is not visited
        """

        # get last vertex that was visited
        last_vertex = visited[-1]

        def distance_to_reference_point(point):
            x, y = point
            # if a^2 >= b^2, then a >= b. Save some time, don't square.
            # modified sqrt( (x_1 - x_2)^2 + (y_1 - y_2)^2 ) to save computation.
            return abs(x - last_vertex[0]) + abs(y - last_vertex[1])

        distanced_points = sorted(stack, key=distance_to_reference_point)

        stack.clear()
        stack.extend(list(reversed(distanced_points)))

    # use dfs algorithm to follow points
    async def dfs_iterative(initial):
        stack, path = [*initial], []

        init_point = stack.pop()
        path.append(init_point)

        while stack:
            await distance_calculation(stack, path)
            vertex = stack.pop()

            path.append(vertex)

        return path

    return asyncio.run(dfs_iterative(points))


class Oscilloscope:
    def __init__(self, media_path: str, hide_progress: bool = False, is_image: bool = False, duration_s: int = 1):
        # Open the media file
        if is_image:
            self.image = cv2.imread(media_path)
            # Check if the video file was opened successfully
            if not (type(self.image) is np.ndarray):
                raise IOError("Failed to open the image file.")

            self.frame_width = int(self.image.shape[1] * .5)
            self.frame_height = int(self.image.shape[0] * .5)
            self.duration = duration_s
        else:
            self.video = cv2.VideoCapture(media_path)
            # Check if the video file was opened successfully
            if not self.video.isOpened():
                raise IOError("Failed to open the video file.")

            fps = self.video.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
            self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH) * .5)  # reduced width (50%)
            self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) * .5)  # reduced height(50%)
            self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_duration = 1 / fps  # duration per frame (used for speeding up playback of audio)

        # flags
        self.is_image: bool = is_image
        self.hide_progress: bool = hide_progress

    def convert(self):
        # clear output file
        with open('out.wav', "w") as file:
            write('out.wav', 44100, np.array([[0, 0], [0, 0]]))
            file.close()

        if self.is_image:
            path = self.__detect_path(self.image)

            # convert the frame to audio
            self.__convert_to_audio(path)

            return

        if self.hide_progress:
            # Iterate over frames without progress bar
            while self.video.isOpened():
                # Read the next frame from the video file
                eof, frame = self.video.read()

                # Check if we reached the end of the video file
                if not eof:
                    break

                # Run path detection on the frame
                path = self.__detect_path(frame)

                # convert the frame to audio
                self.__convert_to_audio(path)
        else:
            # Iterate over the frames of the video file
            with tqdm(total=self.frame_count, unit="frames") as pbar:
                while self.video.isOpened():
                    # Read the next frame from the video file
                    eof, frame = self.video.read()

                    # Check if we reached the end of the video file
                    if not eof:
                        break

                    # Run path detection on the frame
                    path = self.__detect_path(frame)

                    # convert the frame to audio
                    self.__convert_to_audio(path)

                    # update progress bar
                    pbar.update(1)

        # Release the video capture object and close audio file
        self.video.release()

    def __detect_path(self, frame):
        # dsize
        dsize = (self.frame_width, self.frame_height)

        # resize image
        frame_resized = cv2.resize(frame, dsize)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # remove noise
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)
        out_gray = cv2.divide(gray, bg, scale=255)
        out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

        # Run Canny edge detection on the frame
        edges = cv2.Canny(out_binary, 100, 200)

        # Get the indices of non-zero elements
        y_indices, x_indices = np.nonzero(edges)

        # Combine the x and y indices into a list of tuples
        path = list(zip(x_indices, y_indices))

        if not path:
            path = [(0, 0)]

        sorted_path = sort_algorithm(path)

        return sorted_path

    def __convert_to_audio(self, path):
        # encode path as audio

        left_sample = []
        right_sample = []

        # apply transformations
        for i in range(len(path)):
            x, y = path[i]
            left_sample.append(((x - (self.frame_width / 2)) / (self.frame_width / 2)))
            right_sample.append(-((y - (self.frame_height / 2)) / (self.frame_height / 2)))

        left_wave = np.asarray(left_sample)
        right_wave = np.asarray(right_sample)

        stereo_waveform = np.column_stack([left_wave, right_wave])

        sample_rate, data = read('out.wav')

        if self.is_image:
            # speed up audio to make it the duration of 1 frame (at 60fps) -- this is to make it smooth
            # if there are a lot of points it will be slower to make the image.
            target_length = int(sample_rate * 1/60)

            # Create an indices array that represents the position of the samples you want to keep in the new audio data
            indices = np.linspace(0, len(stereo_waveform), target_length, endpoint=False, dtype=int)

            # Use the indices array to select the samples from the original audio data
            frame_data = stereo_waveform[indices]

            # target length to extend/compress waveform
            length_s = len(frame_data) / sample_rate
            num_loops = int(self.duration // length_s)

            # if no conditions then audio length is correct without modifications
            if num_loops < 1:
                # loop audio (extend waveform via duplication)

                frame_data = np.tile(frame_data, (num_loops, 1))
            elif num_loops > 1:
                # speed up audio (compress waveform)
                target_length = int(sample_rate * self.duration)

                indices = np.linspace(0, len(frame_data), target_length, endpoint=False, dtype=int)

                frame_data = frame_data[indices]
        else:
            # speed up audio to make it the duration of 1 frame
            target_length = int(sample_rate * self.frame_duration)

            # Create an indices array that represents the position of the samples you want to keep in the new audio data
            indices = np.linspace(0, len(stereo_waveform), target_length, endpoint=False, dtype=int)

            # Use the indices array to select the samples from the original audio data
            frame_data = stereo_waveform[indices]

        data = np.concatenate((data, frame_data))
        write('out.wav', 44100, data)
