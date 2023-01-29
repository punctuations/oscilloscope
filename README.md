## Oscilloscope

This is a open source converter for video/image formats
to be converted to audio, specifically audio that can be played
back on an oscilloscope in X/Y mode (or on a [simulator](https://dood.al/oscilloscope/))
so that the original media will once again be displayed

### Overview
- [Why make this?](#why)
- [Intentions](#intentions)
- [How it works](#how-it-works)
- [Algorithm](#algorithm)
- [Examples](#examples)


#### Why?

This was a project originally made for a class, post-presentation I cleaned it up to
put it here :)

<sup>[**Back to overview**](#overview)</sup>

#### Intentions

The whole reason for this being open source is so that hopefully if anyone wants to do something in similar they have something to go off of.
I fully recognize this project could be improved upon, especially in it's efficiency as generation times tends to grow a lot as more points are added.

The time-complexity of this is best represented through O(n log n)
-- (Where n is equal to the number of points)
![](https://mattjmatthias.co/content/images/big-o-chart.png)

formally,
![](examples/time_complexity.png)

<sup>[**Back to overview**](#overview)</sup>

#### How it works

The converter works by taking in a path to an image or video
and opens it using python-opencv, to convert it into a numpy array.

After this compression and noise reduction are applied to limit the available points,
this then undergoes a path calculation on it to draw an outline of the frame/image.

The now edge map will be processed as (x, y) points on a cartesian plane and treated as such
from here on.

A reference point is assigned (this will be the last point in the edge map)
and the a modified DFS algorithm will sort all the other points by how close it is
in proximity to the reference point. This process is repeated.

The list of all visited (sorted) points then have translations applied to them to get it in terms of -1.0 <= y <= 1.0
(amplitude of a wave) and they are then recombined into a stereo waveform using numpy's column stack.

After this conversion to audio data, depending on the media form, the points will sped up to reach the correct frame rate to
smooth out the playback when there are a lot of points, and for images the duration passed into the class will be taken into account
and audio data will be looped to reach this duration (making a solid image displayed for _x_ seconds).

<sup>[**Back to overview**](#overview)</sup>

#### Algorithm

This project uses a modified DFS algorithm to accomplish a clean sort of points,
as in traditional DFS algorithms most points have defined neighbours, whereas in this case they do not, as disconnected points
can exist.

The algorithm starts very similar to a normal one except for that the stack (queue)
will be sorted by the algorithm everytime, to compensate for the loss of neighbours.

Every iteration of the algorithm it will get all x and y values and compute distance (using pythagoras adapted for distance)
away from the reference point it will set the queue to this new sorted list.

Below is a picture of the pythagoras distance calculation:

![](examples/distance.png)

<sup>[**Back to overview**](#overview)</sup>

## Examples

To break it down a little farther you can use the example file `examples/circle.py` to generate a cosine and sine wave in the left and right channel,
which when in X/Y mode, will create a circle.

For a visual representation see the diagram below.

![](https://camo.githubusercontent.com/3399db6e0f1e9ee92258131bd5131031c225db4be54d56bf2cb21ec9f472c48d/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f622f62302f4c697373616a6f75735f666967757265735f6f6e5f6f7363696c6c6f73636f70655f25323839305f646567726565735f70686173655f73686966742532392e676966)

<sup>[**Back to overview**](#overview)</sup>
