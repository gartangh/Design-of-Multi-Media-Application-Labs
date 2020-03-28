#!/usr/bin/env python3

import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gtk', '3.0')
gi.require_version('GdkX11', '3.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, Gtk, GLib, GdkX11, GstVideo


class Player(object):

    def __init__(self):
        # initialize GTK
        Gtk.init(sys.argv)
        # initialize GStreamer
        Gst.init(sys.argv)

        self.state = Gst.State.NULL
        self.duration = Gst.CLOCK_TIME_NONE

        # create empty pipeline
        self.pipeline = Gst.Pipeline.new("test-pipeline")
        if not self.pipeline:
            print("ERROR: Could not create pipeline")
            sys.exit(1)

        # create the elements
        # FILESRC 1
        self.source1 = Gst.ElementFactory.make("filesrc", "source1")
        self.demuxer1 = Gst.ElementFactory.make("qtdemux", "demuxer1")
        self.queueD1 = Gst.ElementFactory.make("queue", "queueD1")
        self.decoder1 = Gst.ElementFactory.make("avdec_h264", "decoder1")
        self.queue1 = Gst.ElementFactory.make("queue", "queue1")
        self.videoscale1 = Gst.ElementFactory.make("videoscale", "videoscale1")
        if not self.source1 or not self.demuxer1 or not self.queueD1 or not self.decoder1 or not self.queue1 or not self.videoscale1:
            print("ERROR: Could not create elements from FILESRC 1")
            sys.exit(1)
        self.source1.set_property("location", "sintel_SD.mp4")
        # FILESRC 2
        self.source2 = Gst.ElementFactory.make("filesrc", "source2")
        self.demuxer2 = Gst.ElementFactory.make("qtdemux", "demuxer2")
        self.queueD2 = Gst.ElementFactory.make("queue", "queueD2")
        self.decoder2 = Gst.ElementFactory.make("avdec_h264", "decoder2")
        self.queue2 = Gst.ElementFactory.make("queue", "queue2")
        self.videoscale2 = Gst.ElementFactory.make("videoscale", "videoscale2")
        if not self.source2 or not self.demuxer2 or not self.queueD2 or not self.decoder2 or not self.queue2 or not self.videoscale2:
            print("ERROR: Could not create elements from FILESRC 2")
            sys.exit(1)
        self.source2.set_property("location", "sita_SD.mp4")
        # VIDEOMIXER
        self.videomixer = Gst.ElementFactory.make("videomixer", "videomixer")
        self.videoconvert1 = Gst.ElementFactory.make("videoconvert", "videoconvert1")
        if not self.videomixer or not self.videoconvert1:
            print("ERROR: Could not create elements from VIDEOMIXER")
            sys.exit(1)
        self.videomixer.get_request_pad("sink_0").set_property("alpha", 0.5)
        self.videomixer.get_request_pad("sink_1").set_property("alpha", 0.5)
        # TEE
        self.tee = Gst.ElementFactory.make("tee", "tee")
        if not self.tee:
            print("ERROR: Could not create elements from TEE")
            sys.exit(1)
        # TEE VIDEOSINK
        self.queueA = Gst.ElementFactory.make("queue", "queueA")
        self.videoconvert2 = Gst.ElementFactory.make("videoconvert", "videoconvert2")
        self.ximagesink = Gst.ElementFactory.make("ximagesink", "ximagesink")
        if not self.queueA or not self.ximagesink:
            print("ERROR: Could not create elements from TEE VIDEOSINK")
            sys.exit(1)
        # TEE FILESINK
        self.queueB = Gst.ElementFactory.make("queue", "queueB")
        self.x264enc = Gst.ElementFactory.make("x264enc", "x264enc")
        self.h264parse = Gst.ElementFactory.make("h264parse", "h264parse")
        self.queueB2 = Gst.ElementFactory.make("queue", "queueB2")
        self.matroskamux = Gst.ElementFactory.make("matroskamux", "matroskamux")
        self.filesink = Gst.ElementFactory.make("filesink", "filesink")
        if not self.queueB or not self.x264enc or not self.h264parse or not self.queueB2 or not self.matroskamux or not self.filesink:
            print("ERROR: Could not create elements from TEE FILESINK")
            sys.exit(1)
        self.x264enc.set_property("tune", "zerolatency")
        self.filesink.set_property("location", "output.mkv")

        # add elements to pipeline
        # ADD FILESRC 1
        self.pipeline.add(self.source1)
        self.pipeline.add(self.demuxer1)
        self.pipeline.add(self.queueD1)
        self.pipeline.add(self.decoder1)
        self.pipeline.add(self.queue1)
        self.pipeline.add(self.videoscale1)
        # ADD FILESRC 2
        self.pipeline.add(self.source2)
        self.pipeline.add(self.demuxer2)
        self.pipeline.add(self.queueD2)
        self.pipeline.add(self.decoder2)
        self.pipeline.add(self.queue2)
        self.pipeline.add(self.videoscale2)
        # ADD VIDEOMIXER
        self.pipeline.add(self.videomixer)
        self.pipeline.add(self.videoconvert1)
        # ADD TEE
        self.pipeline.add(self.tee)
        # ADD TEE VIDEOSINK
        self.pipeline.add(self.queueA)
        self.pipeline.add(self.videoconvert2)
        self.pipeline.add(self.ximagesink)
        # ADD TEE FILESINK
        self.pipeline.add(self.queueB)
        self.pipeline.add(self.x264enc)
        self.pipeline.add(self.h264parse)
        self.pipeline.add(self.queueB2)
        self.pipeline.add(self.matroskamux)
        self.pipeline.add(self.filesink)

        # link elements in pipeline
        # LINK FILESRC 1
        if not self.source1.link(self.demuxer1):
            print("ERROR: Could not link 'source1' to 'demuxer1'")
            sys.exit(1)
        # dynamic link between demuxer1 and queueD1
        if not self.queueD1.link(self.decoder1):
            print("ERROR: Could not link 'queueD1' to 'decoder1'")
            sys.exit(1)
        if not self.decoder1.link(self.queue1):
            print("ERROR: Could not link 'decoder1' to 'queue1'")
            sys.exit(1)
        if not self.queue1.link(self.videoscale1):
            print("ERROR: Could not link 'queue1' to 'videoscale1'")
            sys.exit(1)
        # LINK FILESRC 2
        if not self.source2.link(self.demuxer2):
            print("ERROR: Could not link 'source2' to 'demuxer2'")
            sys.exit(1)
        # dynamic link between demuxer2 and queueD2
        if not self.queueD2.link(self.decoder2):
            print("ERROR: Could not link 'queueD2' to 'decoder2'")
            sys.exit(1)
        if not self.decoder2.link(self.queue2):
            print("ERROR: Could not link 'decoder2' to 'queue2'")
            sys.exit(1)
        if not self.queue2.link(self.videoscale2):
            print("ERROR: Could not link 'queue2' to 'videoscale2'")
            sys.exit(1)
        # LINK VIDEOMIXER
        if not self.videoscale1.link(self.videomixer):
            print("ERROR: Could not link 'videoscale1' to 'videomixer'")
            sys.exit(1)
        if not self.videoscale2.link(self.videomixer):
            print("ERROR: Could not link 'videoscale2' to 'videomixer'")
            sys.exit(1)
        if not self.videomixer.link(self.videoconvert1):
            print("ERROR: Could not link 'videomixer' to 'videoconvert1'")
            sys.exit(1)
        # LINK TEE
        if not self.videoconvert1.link(self.tee):
            print("ERROR: Could not link 'videoconvert1' to 'tee'")
            sys.exit(1)
        # LINK TEE VIDEOSINK
        if not self.tee.link(self.queueA):
            print("ERROR: Could not link 'tee' to 'queueA'")
            sys.exit(1)
        if not self.queueA.link(self.videoconvert2):
            print("ERROR: Could not link 'queueA' to 'videoconvert2'")
            sys.exit(1)
        if not self.videoconvert2.link(self.ximagesink):
            print("ERROR: Could not link 'videoconvert2' to 'ximagesink'")
            sys.exit(1)
        # LINK TEE FILESINK
        if not self.tee.link(self.queueB):
            print("ERROR: Could not link 'tee' to 'queueB'")
            sys.exit(1)
        if not self.queueB.link(self.x264enc):
            print("ERROR: Could not link 'queueB' to 'x264enc'")
            sys.exit(1)
        if not self.x264enc.link(self.h264parse):
            print("ERROR: Could not link 'x264enc' to 'h264parse'")
            sys.exit(1)
        if not self.h264parse.link(self.queueB2):
            print("ERROR: Could not link 'h264parse' to 'queueB2'")
            sys.exit(1)
        if not self.queueB2.link(self.matroskamux):
            print("ERROR: Could not link 'queueB2' to 'matroskamux'")
            sys.exit(1)
        if not self.matroskamux.link(self.filesink):
            print("ERROR: Could not link 'matroskamux' to 'filesink'")
            sys.exit(1)
        
        # dynamic link between demuxer1 and queue1
        self.demuxer1.connect("pad-added", self.on_pad_added_demuxer1)
        # dynamic link between demuxer2 and queue2
        self.demuxer2.connect("pad-added", self.on_pad_added_demuxer2)

        # set initial alpha values
        #self.videomixer.get_request_pad("sink_0").set_property("alpha", 0.5)
        #self.videomixer.get_request_pad("sink_1").set_property("alpha", 0.5)

        # start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline set to playing")
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Unable to set the pipeline to the playing state")
            sys.exit(1)

        # create the GUI
        self.build_ui()

        # instruct the bus to emit signals for each received message
        # and connect to the interesting signals
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self.on_error)
        bus.connect("message::eos", self.on_eos)
        bus.connect("message::state-changed", self.on_state_changed)
        bus.connect("message::application", self.on_application_message)

    # set the pipeline to PLAYING (start playback), register refresh callback
    # and start the GTK main loop
    def start(self):
        # start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline set to playing")
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Unable to set the pipeline to the playing state")
            sys.exit(1)

        # register a function that GLib will call every second
        #GLib.timeout_add_seconds(1, self.refresh_ui)

        # start the GTK main loop. we will not regain control until
        # Gtk.main_quit() is called
        Gtk.main()

        # free resources
        self.cleanup()

    # set the pipeline state to NULL and remove the reference to it
    def cleanup(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None

    def build_ui(self):
        main_window = Gtk.Window.new(Gtk.WindowType.TOPLEVEL)
        main_window.connect("delete-event", self.on_delete_event)

        video_window = Gtk.DrawingArea.new()
        video_window.set_double_buffered(False)
        video_window.connect("realize", self.on_realize)
        video_window.connect("draw", self.on_draw)

        #play_button = Gtk.Button.new_from_stock(Gtk.STOCK_MEDIA_PLAY)
        #play_button.connect("clicked", self.on_play)

        #pause_button = Gtk.Button.new_from_stock(Gtk.STOCK_MEDIA_PAUSE)
        #pause_button.connect("clicked", self.on_pause)

        #stop_button = Gtk.Button.new_from_stock(Gtk.STOCK_MEDIA_STOP)
        #stop_button.connect("clicked", self.on_stop)

        self.slider = Gtk.HScale.new_with_range(0, 100, 1)
        self.slider.set_draw_value(False)
        self.slider.connect("value-changed", self.on_slider_changed)
        self.slider.set_value(50)

        self.streams_list = Gtk.TextView.new()
        self.streams_list.set_editable(False)

        controls = Gtk.HBox.new(False, 0)
        #controls.pack_start(play_button, False, False, 2)
        #controls.pack_start(pause_button, False, False, 2)
        #controls.pack_start(stop_button, False, False, 2)
        controls.pack_start(self.slider, True, True, 0)

        main_hbox = Gtk.HBox.new(False, 0)
        main_hbox.pack_start(video_window, True, True, 0)
        main_hbox.pack_start(self.streams_list, False, False, 2)

        main_box = Gtk.VBox.new(False, 0)
        main_box.pack_start(main_hbox, True, True, 0)
        main_box.pack_start(controls, False, False, 0)

        main_window.add(main_box)
        main_window.set_default_size(640, 480)
        main_window.show_all()

    # this function is called when the GUI toolkit creates the physical window
    # that will hold the video. At this point we can retrieve its handler and pass
    # it to GStreamer
    def on_realize(self, widget):
        window = widget.get_window()
        window_handle = window.get_xid()

        # pass it to pipeline, which will forward it to the video sink
        self.ximagesink.set_window_handle(window_handle)

    # this function is called when the PLAY button is clicked
    def on_play(self, button):
        self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline set to playing")
        pass

    # this function is called when the PAUSE button is clicked
    def on_pause(self, button):
        self.pipeline.set_state(Gst.State.PAUSED)
        pass

    # this function is called when the STOP button is clicked
    def on_stop(self, button):
        self.pipeline.set_state(Gst.State.READY)
        pass

    # this function is called when the main window is closed
    def on_delete_event(self, widget, event):
        self.on_stop(None)
        Gtk.main_quit()

    # this function is called every time the video window needs to be
    # redrawn. GStreamer takes care of this in the PAUSED and PLAYING states.
    # in the other states we simply draw a black rectangle to avoid
    # any garbage showing up
    def on_draw(self, widget, cr):
        if self.state < Gst.State.PAUSED:
            allocation = widget.get_allocation()

            cr.set_source_rgb(0, 0, 0)
            cr.rectangle(0, 0, allocation.width, allocation.height)
            cr.fill()

        return False

    # this function is called when the slider changes its position.
    # we perform a seek to the new position here
    def on_slider_changed(self, range):
        value = self.slider.get_value()/100
        print("slider value changed: " + str(value))
        self.videomixer.get_static_pad("sink_0").set_property("alpha", 1-value)
        self.videomixer.get_static_pad("sink_1").set_property("alpha", value)

    # this function is called when new metadata is discovered in the stream
    def on_tags_changed(self, pipeline, stream):
        # we are possibly in a GStreamer working thread, so we notify
        # the main thread of this event through a message in the bus
        self.pipeline.post_message(
            Gst.Message.new_application(
                self.pipeline,
                Gst.Structure.new_empty("tags-changed")))

    # this function is called when an error message is posted on the bus
    def on_error(self, bus, msg):
        err, dbg = msg.parse_error()
        print("ERROR:", msg.src.get_name(), ":", err.message)
        if dbg:
            print("Debug info:", dbg)

    # this function is called when an End-Of-Stream message is posted on the bus
    # we just set the pipeline to READY (which stops playback)
    def on_eos(self, bus, msg):
        print("End-Of-Stream reached")
        self.pipeline.set_state(Gst.State.READY)

    # this function is called when the pipeline changes states.
    # we use it to keep track of the current state
    def on_state_changed(self, bus, msg):
        old, new, pending = msg.parse_state_changed()
        if not msg.src == self.pipeline:
            # not from the pipeline, ignore
            return

        self.state = new
        print("State changed from {0} to {1}".format(Gst.Element.state_get_name(old), Gst.Element.state_get_name(new)))

        if old == Gst.State.READY and new == Gst.State.PAUSED:
            # for extra responsiveness we refresh the GUI as soons as
            # we reach the PAUSED state
            pass

    # extract metadata from all the streams and write it to the text widget
    # in the GUI
    def analyze_streams(self):
        # clear current contents of the widget
        buffer = self.streams_list.get_buffer()
        buffer.set_text("")

        # read some properties
        nr_video = self.pipeline.get_property("n-video")
        nr_audio = self.pipeline.get_property("n-audio")
        nr_text = self.pipeline.get_property("n-text")

        for i in range(nr_video):
            tags = None
            # retrieve the stream's video tags
            tags = self.pipeline.emit("get-video-tags", i)
            if tags:
                buffer.insert_at_cursor("video stream {0}\n".format(i))
                _, str = tags.get_string(Gst.TAG_VIDEO_CODEC)
                buffer.insert_at_cursor(
                    "  codec: {0}\n".format(
                        str or "unknown"))

        for i in range(nr_audio):
            tags = None
            # retrieve the stream's audio tags
            tags = self.pipeline.emit("get-audio-tags", i)
            if tags:
                buffer.insert_at_cursor("\naudio stream {0}\n".format(i))
                ret, str = tags.get_string(Gst.TAG_AUDIO_CODEC)
                if ret:
                    buffer.insert_at_cursor(
                        "  codec: {0}\n".format(
                            str or "unknown"))

                ret, str = tags.get_string(Gst.TAG_LANGUAGE_CODE)
                if ret:
                    buffer.insert_at_cursor(
                        "  language: {0}\n".format(
                            str or "unknown"))

                ret, str = tags.get_uint(Gst.TAG_BITRATE)
                if ret:
                    buffer.insert_at_cursor(
                        "  bitrate: {0}\n".format(
                            str or "unknown"))

        for i in range(nr_text):
            tags = None
            # retrieve the stream's subtitle tags
            tags = self.pipeline.emit("get-text-tags", i)
            if tags:
                buffer.insert_at_cursor("\nsubtitle stream {0}\n".format(i))
                ret, str = tags.get_string(Gst.TAG_LANGUAGE_CODE)
                if ret:
                    buffer.insert_at_cursor(
                        "  language: {0}\n".format(
                            str or "unknown"))

    # this function is called when an "application" message is posted on the bus
    # here we retrieve the message posted by the on_tags_changed callback
    def on_application_message(self, bus, msg):
        if msg.get_structure().get_name() == "tags-changed":
            # if the message is the "tags-changed", update the stream info in
            # the GUI
            self.analyze_streams()

    # handler for the pad-added signal
    def on_pad_added_demuxer1(self, src, new_pad):
        #print("Received new pad '{0:s}' from '{1:s}'".format(new_pad.get_name(), src.get_name()))

        # check the new pad's type
        new_pad_caps = new_pad.get_current_caps()
        new_pad_struct = new_pad_caps.get_structure(0)
        new_pad_type = new_pad_struct.get_name()

        if new_pad_type.startswith("video/x-h264"):
            sink_pad = self.queueD1.get_static_pad("sink")
        else:
            #print("It has type '{0:s}' which is not video. Ignoring.".format(new_pad_type))
            return

        # if our converter is already linked, we have nothing to do here
        if sink_pad.is_linked():
            #print("We are already linked. Ignoring.")
            return

        # attempt the link
        ret = new_pad.link(sink_pad)
        if not ret == Gst.PadLinkReturn.OK:
            print("Type is '{0:s}' but link failed".format(new_pad_type))
        else:
            #print("Link succeeded (type '{0:s}')".format(new_pad_type))
            pass

    def on_pad_added_demuxer2(self, src, new_pad):
        #print("Received new pad '{0:s}' from '{1:s}'".format(new_pad.get_name(), src.get_name()))

        # check the new pad's type
        new_pad_caps = new_pad.get_current_caps()
        new_pad_struct = new_pad_caps.get_structure(0)
        new_pad_type = new_pad_struct.get_name()

        if new_pad_type.startswith("video/x-h264"):
            sink_pad = self.queueD2.get_static_pad("sink")
        else:
            #print("It has type '{0:s}' which is not video. Ignoring.".format(new_pad_type))
            return

        # if our converter is already linked, we have nothing to do here
        if sink_pad.is_linked():
            #print("We are already linked. Ignoring.")
            return

        # attempt the link
        ret = new_pad.link(sink_pad)
        if not ret == Gst.PadLinkReturn.OK:
            print("Type is '{0:s}' but link failed".format(new_pad_type))
        else:
            #print("Link succeeded (type '{0:s}')".format(new_pad_type))
            pass

if __name__ == '__main__':
    p = Player()
    p.start()
