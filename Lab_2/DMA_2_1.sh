# clear screen
clear >$(tty)

# files
#SINTEL="sintel_SD.mp4"
SINTEL="http://users.datasciencelab.ugent.be/MM/sintel_SD.mp4"
#gst-discoverer-1.0 $SINTEL
#SITA="sita_SD.mp4"
SITA="http://users.datasciencelab.ugent.be/MM/sita_SD.mp4"
#gst-discoverer-1.0 $SITA

gst-launch-1.0 videomixer name=mix sink_0::alpha=0.5 sink_1::alpha=0.5 \
    ! solarize ! videoconvert ! tee name=t \
	   ! queue max-size-time=0 max-size-buffers=0 max-size-bytes=0 ! videoconvert ! ximagesink \
	t. ! queue ! x264enc ! h264parse ! queue ! matroskamux ! filesink location=mix.mkv \
    souphttpsrc location=$SINTEL ! qtdemux name=d1 d1.video_0 ! queue ! avdec_h264 ! queue ! videoscale \
            ! video/x-raw, width=640, height=480 \
            ! mix.sink_0 \
    souphttpsrc location=$SITA   ! qtdemux name=d2 d2.video_0 ! queue ! avdec_h264 ! queue ! videoscale \
            ! video/x-raw, width=640, height=480 \
            ! mix.sink_1