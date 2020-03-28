#!/usr/bin/env python3

# https://gstreamer.freedesktop.org/documentation/tutorials/basic/hello-world.html

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib

# initialize GStreamer
Gst.init(None)

# build the pipeline
pipeline = Gst.parse_launch(
    "playbin uri=http://users.datasciencelab.ugent.be/MM/sintel_SD.mp4"
)

# start playing
pipeline.set_state(Gst.State.PLAYING)

# wait until EOS or error
bus = pipeline.get_bus()
msg = bus.timed_pop_filtered(
    Gst.CLOCK_TIME_NONE,
    Gst.MessageType.ERROR | Gst.MessageType.EOS
)

# free resources
pipeline.set_state(Gst.State.NULL)
