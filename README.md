## Oscilloscope

This is a open source converter for video/image formats to be converted to
audio, specifically audio that can be played back on an oscilloscope in X/Y mode
(or on a [simulator](https://dood.al/oscilloscope/)) so that the original media
will once again be displayed

### Overview

- [Why make this?](#why)
- [Intentions](#intentions)
- [How it works](#how-it-works)
- [Algorithm](#algorithm)
- [Examples](#examples)
- [Similar Projects](#similar-projects)

### CLI

You can run the converter as a CLI:

`python -m oscilloscope <path> [flags]`

Examples:

- Video:
  - `python -m oscilloscope input.mp4 -o out.wav --workers 6 --max-points 12000`
- Image (auto-detected by extension):
  - `python -m oscilloscope input.png -o out.wav --duration 3`
- Force mode:
  - `python -m oscilloscope input.dat --image --duration 2`

Useful flags:

- `--no-progress` disables the progress bar
- `--max-points` caps work per frame
- `--workers` / `--prefetch` control concurrent video processing
- `--contours` enables contour-based paths (optional)
- `--no-contours` (default) uses edge-pixels + point ordering
- `--ordering` controls point ordering (`kd` default, `kd-flat`, `morton`)
- `--skip-solid` (default) writes silence for solid-color frames; adjust with
  `--solid-threshold`
- `--scale` is the biggest speed lever (lower = faster)
- `--fast` uses a faster contour-tracing path (often faster and looks clean)

#### Why?

This was a project originally made for a class, post-presentation I cleaned it
up to put it here :)

<sup>[**Back to overview**](#overview)</sup>

#### Intentions

The whole reason for this being open source is so that hopefully if anyone wants
to do something in similar they have something to go off of. I fully recognize
this project is extremely niche in use but I figured why not.

<sup>[**Back to overview**](#overview)</sup>

#### How it works

The converter works by taking in a path to an image or video and opens it using
python-opencv, to convert it into a numpy array.

After this, light noise reduction is applied and an edge map is extracted.

The edge map is processed as (x, y) points on a cartesian plane and treated as
such from here on.

From there, points are ordered into a path that an oscilloscope can trace.
Disconnected shapes are handled explicitly to avoid the path constantly jumping
between separate outlines (which creates faint connector lines).

The ordered points are normalized into the range [-1, 1] and combined into a
stereo waveform (X on left channel, Y on right channel).

Audio is written as a streaming WAV file (so it doesn’t re-read/rewrite the
whole file each frame). For videos, frames are processed concurrently and
written to WAV in order.

<sup>[**Back to overview**](#overview)</sup>

#### Algorithm

At a high level the pipeline is:

1. Preprocess frame (resize + grayscale + optional blur/noise cleanup)
2. Extract edges (Canny)
3. Convert edges to a draw path
4. Resample to the target audio frame duration
5. Write PCM samples to a WAV stream

Path construction has a few modes:

- Default (`--no-contours`, `--ordering kd`):
  - Split the edge map into connected components (each component ≈ one shape)
  - Downsample points per component to stay within `--max-points`
  - Build a KD-tree per component and precompute a fixed k-nearest-neighbor
    table
  - Do a greedy nearest-neighbor walk over that cached neighbor table
  - Chain components by nearest endpoints to minimize travel lines between
    shapes

- `--ordering kd-flat`:
  - KD ordering over the full edge-point cloud (can interleave shapes)

- `--ordering morton`:
  - Locality-preserving Morton/Z-order sort (fast, different “stroke” feel)

- `--fast`:
  - Trace contours directly from the edge map and chain them by endpoint
    proximity
  - Avoids KD ordering entirely (often faster)

Time complexity is typically dominated by point ordering. For KD ordering, the
dominant term is $O(n \log n)$ where $n$ is the number of kept edge points
(bounded by `--max-points`), plus $O(W\cdot H)$ image operations on the scaled
frame.

<sup>[**Back to overview**](#overview)</sup>

## Examples

To break it down a little farther you can use the example file
`examples/circle.py` to generate a cosine and sine wave in the left and right
channel, which when in X/Y mode, will create a circle.

For a visual representation see the diagram below.

![](https://camo.githubusercontent.com/3399db6e0f1e9ee92258131bd5131031c225db4be54d56bf2cb21ec9f472c48d/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f622f62302f4c697373616a6f75735f666967757265735f6f6e5f6f7363696c6c6f73636f70655f25323839305f646567726565735f70686173655f73686966742532392e676966)

<sup>[**Back to overview**](#overview)</sup>

### Similar Projects

While I am not the first to do something like this, I believe I am to make it
for video/images, as one (and in python).

Some projects that are similar and I used as reference can be found below:

- [Yeonzi's Bad Apple Oscilloscope](https://github.com/yeonzi/badappe_oscilloscope)
  (C)
- [YJBeetle's OscilloscopePlayer](https://github.com/YJBeetle/OscilloscopePlayer)
  (C++)

<sup>[**Back to overview**](#overview)</sup>
