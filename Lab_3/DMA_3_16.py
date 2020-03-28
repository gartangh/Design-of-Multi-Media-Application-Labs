from __future__ import print_function
import sys
import cv2
import numpy
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
numpy.set_printoptions(threshold=sys.maxsize)

# globals
ROI = None
input_buffer = []
groundtruth_buffer = []
output_buffer = None


# readers
class Reader:
	"""A reader's base class
	"""
	def read(self, filename):
		raise NotImplementedError

class ROIReader(Reader):
	"""Reads the ROI
	"""
	def read(self, filename):
		global ROI

		ROI = cv2.imread(filename,0)

class InputReader(Reader):
	"""Reads the input
	"""
	def read(self, filename):
		# gst-launch-1.0 filesrc location=~/Downloads/DMA_3_video/nightvideo/streetcornernight/input.mkv ! decodebin ! videoconvert ! video/x-raw,format=GRAY8 ! autovideosink
		# here appsrc instead of autovideosink

		# create pipeline
		self.source = Gst.ElementFactory.make("filesrc", "source")
		self.decode = Gst.ElementFactory.make("decodebin", "decode")
		self.convert = Gst.ElementFactory.make("videoconvert", "convert")
		self.sink = Gst.ElementFactory.make("appsink", "sink")
		self.pipeline = Gst.Pipeline.new("input_reader_pipeline")
		if not self.source or not self.decode or not self.convert or not self.sink or not self.pipeline:
			print("Not all elements could be created.")
			exit(-1)

		# set properties
		self.source.set_property("location", filename)
		self.sink.set_property("emit-signals", True)
		#self.sink.set_property("max-buffers", 2)
		#self.sink.set_property("drop", True)
		#self.sink.set_property("sync", False)
		caps = Gst.caps_from_string("video/x-raw, format=(string){GRAY8};")
		self.sink.set_property("caps", caps)

		# build pipeline
		self.pipeline.add(self.source)
		self.pipeline.add(self.decode)
		self.pipeline.add(self.convert)
		self.pipeline.add(self.sink)
		if not Gst.Element.link(self.source, self.decode):
			print("Elements could not be linked.")
			exit(-1)
		# dynamic link between decode and convert
		if not Gst.Element.link(self.convert, self.sink):
			print("Elements could not be linked.")
			exit(-1)
		# dynamic link between decode and convert
		self.decode.connect("pad-added", self.on_pad_added)
		# dynamic link between sink and input_buffer
		self.sink.connect("new-sample", self.new_buffer, self.sink)

		# start playing
		ret = self.pipeline.set_state(Gst.State.PLAYING)
		if ret == Gst.StateChangeReturn.FAILURE:
			print("Unable to set the pipeline to the playing state.")
			exit(-1)

		# wait until error or EOS
		bus = self.pipeline.get_bus()

		# parse message
		while True:
			while len(input_buffer) != 0:
				input_frame = input_buffer.pop(0)
				yield input_frame

			message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
		
			if message:
				if message.type == Gst.MessageType.ERROR:
					err, debug = message.parse_error()
					print("Error received from element %s: %s" % (
						message.src.get_name(), err))
					print("Debugging information: %s" % debug)
					break
				elif message.type == Gst.MessageType.EOS:
					print("End-Of-Stream reached.")
					break
				elif message.type == Gst.MessageType.STATE_CHANGED:
					if isinstance(message.src, Gst.Pipeline):
						old_state, new_state, pending_state = message.parse_state_changed()
						print("Pipeline state changed from %s to %s." %
							(old_state.value_nick, new_state.value_nick))
				else:
					print("Unexpected message received.")

		# Free resources
		self.pipeline.set_state(Gst.State.NULL)

	def on_pad_added(self, src, new_pad):
		print("Received new pad '{0:s}' from '{1:s}'".format(new_pad.get_name(), src.get_name()))

		# check the new pad's type
		new_pad_caps = new_pad.get_current_caps()
		new_pad_struct = new_pad_caps.get_structure(0)
		new_pad_type = new_pad_struct.get_name()

		if new_pad_type.startswith("video/x-raw"):
			sink_pad = self.convert.get_static_pad("sink")
		else:
			print("It has type '{0:s}' which is not video. Ignoring.".format(new_pad_type))
			return

		# if our converter is already linked, we have nothing to do here
		if sink_pad.is_linked():
			print("We are already linked. Ignoring.")
			return

		# attempt the link
		ret = new_pad.link(sink_pad)
		if not ret == Gst.PadLinkReturn.OK:
			print("Type is '{0:s}' but link failed".format(new_pad_type))
		else:
			print("Link succeeded (type '{0:s}')".format(new_pad_type))
			pass

	def new_buffer(self, sink, data):
		global input_buffer

		sample = sink.emit("pull-sample")
		input_frame = self.gst_to_opencv(sample)
		input_buffer.append(input_frame)

		return Gst.FlowReturn.OK

	def gst_to_opencv(self, sample):
		buf = sample.get_buffer()
		caps = sample.get_caps()

		#print caps.get_structure(0).get_value('format')
		#print caps.get_structure(0).get_value('height')
		#print caps.get_structure(0).get_value('width')
		#print buf.get_size()

		return numpy.ndarray(
			(caps.get_structure(0).get_value('height'), caps.get_structure(0).get_value('width'), 1),
			buffer=buf.extract_dup(0, buf.get_size()),
			dtype=numpy.uint8)

class GroundtruthReader(Reader):
	"""Reads the groundtruth
	"""
	def read(self, filename):
		# gst-launch-1.0 filesrc location=~/Downloads/DMA_3_video/nightvideo/streetcornernight/groundtruth.mp4 ! decodebin ! videoconvert ! video/x-raw,format=GRAY8 ! autovideosink
		# here appsrc instead of autovideosink

		# create pipeline
		self.source = Gst.ElementFactory.make("filesrc", "source")
		self.decode = Gst.ElementFactory.make("decodebin", "decode")
		self.convert = Gst.ElementFactory.make("videoconvert", "convert")
		self.sink = Gst.ElementFactory.make("appsink", "sink")
		self.pipeline = Gst.Pipeline.new("groundtruth_reader_pipeline")
		if not self.source or not self.decode or not self.convert or not self.sink or not self.pipeline:
			print("Not all elements could be created.")
			exit(-1)

		# set properties
		self.source.set_property("location", filename)
		self.sink.set_property("emit-signals", True)
		# sink.set_property("max-buffers", 2)
		# # sink.set_property("drop", True)
		# # sink.set_property("sync", False)
		caps = Gst.caps_from_string("video/x-raw, format=(string){GRAY8};")
		self.sink.set_property("caps", caps)

		# build pipeline
		self.pipeline.add(self.source)
		self.pipeline.add(self.decode)
		self.pipeline.add(self.convert)
		self.pipeline.add(self.sink)
		if not Gst.Element.link(self.source, self.decode):
			print("Elements could not be linked.")
			exit(-1)
		# dynamic link between decode and convert
		if not Gst.Element.link(self.convert, self.sink):
			print("Elements could not be linked.")
			exit(-1)

		# dynamic link between decode and convert
		self.decode.connect("pad-added", self.on_pad_added)
		# dynamic link between sink and groundtruth_buffer
		self.sink.connect("new-sample", self.new_buffer, self.sink)

		# start playing
		ret = self.pipeline.set_state(Gst.State.PLAYING)
		if ret == Gst.StateChangeReturn.FAILURE:
			print("Unable to set the pipeline to the playing state.")
			exit(-1)

		# wait until error or EOS
		bus = self.pipeline.get_bus()

		# parse message
		while True:
			while len(groundtruth_buffer) != 0:
				groundtruth_frame = groundtruth_buffer.pop(0)
				yield groundtruth_frame

			message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)

			if message:
				if message.type == Gst.MessageType.ERROR:
					err, debug = message.parse_error()
					print("Error received from element %s: %s" % (
						message.src.get_name(), err))
					print("Debugging information: %s" % debug)
					break
				elif message.type == Gst.MessageType.EOS:
					print("End-Of-Stream reached.")
					break
				elif message.type == Gst.MessageType.STATE_CHANGED:
					if isinstance(message.src, Gst.Pipeline):
						old_state, new_state, pending_state = message.parse_state_changed()
						print("Pipeline state changed from %s to %s." %
							(old_state.value_nick, new_state.value_nick))
				else:
					print("Unexpected message received.")

		# Free resources
		self.pipeline.set_state(Gst.State.NULL)

	def on_pad_added(self, src, new_pad):
		print("Received new pad '{0:s}' from '{1:s}'".format(new_pad.get_name(), src.get_name()))

		# check the new pad's type
		new_pad_caps = new_pad.get_current_caps()
		new_pad_struct = new_pad_caps.get_structure(0)
		new_pad_type = new_pad_struct.get_name()

		if new_pad_type.startswith("video/x-raw"):
			sink_pad = self.convert.get_static_pad("sink")
		else:
			print("It has type '{0:s}' which is not video. Ignoring.".format(new_pad_type))
			return

		# if our converter is already linked, we have nothing to do here
		if sink_pad.is_linked():
			print("We are already linked. Ignoring.")
			return

		# attempt the link
		ret = new_pad.link(sink_pad)
		if not ret == Gst.PadLinkReturn.OK:
			print("Type is '{0:s}' but link failed".format(new_pad_type))
		else:
			print("Link succeeded (type '{0:s}')".format(new_pad_type))
			pass

	def new_buffer(self, sink, data):
		global groundtruth_buffer

		sample = sink.emit("pull-sample")
		groundtruth_frame = self.gst_to_opencv(sample)
		groundtruth_buffer.append(groundtruth_frame)

		return Gst.FlowReturn.OK

	def gst_to_opencv(self, sample):
		buf = sample.get_buffer()
		caps = sample.get_caps()

		#print caps.get_structure(0).get_value('format')
		#print caps.get_structure(0).get_value('height')
		#print caps.get_structure(0).get_value('width')
		#print buf.get_size()

		return numpy.ndarray(
			(caps.get_structure(0).get_value('height'), caps.get_structure(0).get_value('width'), 1),
			buffer=buf.extract_dup(0, buf.get_size()),
			dtype=numpy.uint8)

# writers
class Writer:
	"""A writer's base class
	"""
	def write(self):
		raise NotImplementedError

class FileWriter(Writer):
	"""Writes the output to a file
	"""
	def write(self, filename):
		# gst-launch-1.0 videotestsrc ! videoconvert ! x264enc ! matroskamux ! filesink location=~/Downloads/DMA_3_video/nightvideo/streetcornernight/input.mkv
		# here appsrc instead of videotestsrc

		# create pipeline
		# TODO make this an appsrc
		self.source = Gst.ElementFactory.make("videotestsrc", "source")
		self.convert = Gst.ElementFactory.make("videoconvert", "convert")
		self.encode = Gst.ElementFactory.make("x264enc", "encode")
		self.mux = Gst.ElementFactory.make("matroskamux", "mux")
		self.sink = Gst.ElementFactory.make("filesink", "sink")
		self.pipeline = Gst.Pipeline.new("output_writer_pipeline")
		if not self.source or not self.convert or not self.encode or not self.mux or not self.sink or not self.pipeline:
			print("Not all elements could be created.")
			exit(-1)

		# set properties
		#self.source.set_property("emit-signals", True)
		#self.source.set_property("max-buffers", 2)
		#self.source.set_property("drop", True)
		#self.source.set_property("sync", False)
		caps = Gst.caps_from_string("video/x-raw, format=(string){GRAY8};")
		#self.source.set_property("caps", caps)
		self.source.set_property("pattern", 0)
		self.sink.set_property("location", filename)

		# build pipeline
		self.pipeline.add(self.source)
		self.pipeline.add(self.convert)
		self.pipeline.add(self.encode)
		self.pipeline.add(self.mux)
		self.pipeline.add(self.sink)
		if not Gst.Element.link(self.source, self.convert):
			print("Elements could not be linked.")
			exit(-1)
		if not Gst.Element.link(self.convert, self. encode):
			print("Elements coudl not be linked.")
			exit(-1)
		if not Gst.Element.link(self.encode, self.mux):
			print("Elements could not be linked.")
			exit(-1)
		if not Gst.Element.link(self.mux, self.sink):
			print("Elements could not be linked.")
			exit(-1)

		# dynamic link between source and output_buffer
		#self.source.connect("new-sample", self.new_buffer, self.source)

		# start playing
		ret = self.pipeline.set_state(Gst.State.PLAYING)
		if ret == Gst.StateChangeReturn.FAILURE:
			print("Unable to set the pipeline to the playing state.")
			exit(-1)

		# wait until error or EOS
		bus = self.pipeline.get_bus()

		# parse message
		while True:
			message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
			# print "output_buffer: ", output_buffer
			if output_buffer is not None:   
				cv2.imshow("appsink output_buffer", output_buffer)
				# TODO set to 1
				cv2.waitKey(0)
			if message:
				if message.type == Gst.MessageType.ERROR:
					err, debug = message.parse_error()
					print("Error received from element %s: %s" % (
						message.src.get_name(), err))
					print("Debugging information: %s" % debug)
					break
				elif message.type == Gst.MessageType.EOS:
					print("End-Of-Stream reached.")
					break
				elif message.type == Gst.MessageType.STATE_CHANGED:
					if isinstance(message.src, Gst.Pipeline):
						old_state, new_state, pending_state = message.parse_state_changed()
						print("Pipeline state changed from %s to %s." %
							(old_state.value_nick, new_state.value_nick))
				else:
					print("Unexpected message received.")

		# Free resources
		self.pipeline.set_state(Gst.State.NULL)

	def new_buffer(self, source, data):
		global output_buffer

		buf = Gst.Buffer.new_allocate(None, len(data), None)
		assert buf is not None
		buf.fill(0, data)
		buf.pts = buf.dts = int(t * 1e9)
		sample = source.emit("push-buffer", buf)
		output_buffer = self.opencv_to_gst(sample)

		return Gst.FlowReturn.OK

	def opencv_to_gst(self, sample):
		buf = sample.get_buffer()
		caps = sample.get_caps()

		#print caps.get_structure(0).get_value('format')
		#print caps.get_structure(0).get_value('height')
		#print caps.get_structure(0).get_value('width')
		#print buf.get_size()

		return numpy.ndarray(
			(caps.get_structure(0).get_value('height'), caps.get_structure(0).get_value('width'), 1),
			buffer=buf.extract_dup(0, buf.get_size()),
			dtype=numpy.uint8)

class ScreenWriter(Writer):
	"""Writes the output to the screen
	"""
	def write(self, frame):
		global ROI

		# show input with ROI filter
		cv2.imshow('Video', numpy.bitwise_and(numpy.squeeze(frame), ROI))
		cv2.waitKey(1)

# estimators
class Estimator:
	"""An estimator
	"""
	def __init__(self, max_buffer_size, foreground_mask_threshold):
		self._max_buffer_size = max_buffer_size
		self._current_index = 0
		self._buffer = []
		# the foreground mask threshold
		self.foreground_mask_threshold = foreground_mask_threshold
		# the motion threshold (1 %)
		self.motion_threshold = 0.01
		# the input motion as binary values
		self.input_motion_buffer = []
		# the groundtruth motion as binary values
		self.groundtruth_motion_buffer = []
		# the current frame count
		self.frame_count = 0
		# The MOG
#history=max_buffer_size, nmixtures=3, backgroundRatio=0.7
		self.mog = cv2.BackgroundSubtractorMOG(history=max_buffer_size, nmixtures=3, backgroundRatio=0.7)

	def estimate(self, input_frame, groundtruth_frame):
		input_frame = numpy.squeeze(input_frame)
		groundtruth_frame = numpy.squeeze(groundtruth_frame)
		foreground_mask = self.feed(input_frame)
		if foreground_mask is not None:
			#cv2.imshow('groundtruth_frame', groundtruth_frame)
			binarized_groundtruth = self.binarize_groundtruth(groundtruth_frame)
			cv2.imshow('binarized_groundtruth', binarized_groundtruth)
			cv2.waitKey(1)
			self.groundtruth_motion(binarized_groundtruth)
			self.input_motion(foreground_mask)

		self.append_to_buffer(numpy.squeeze(input_frame))

	def append_to_buffer(self, frame):
		# append frame to non-filled buffer
		if len(self._buffer) < self._max_buffer_size:
			self._buffer.append(frame)
		# replace frame in buffer by current frame
		else:
			self._buffer[self._current_index] = frame
			self._current_index = (self._current_index + 1) % self._max_buffer_size

	def feed(self, frame):
		"""Feed the motion detector with a frame and return a binary foreground mask
		"""
		# TODO compute background model
		# TODO write background model with ScreenWriter
		# TODO substract frame from background model
		# TODO if absolute values > foreground_mask_threshold 1, else 0 => binarized foregroundmask
		# TODO return foregroundmask
		raise NotImplementedError

	def input_motion(self, frame):
		"""For each input frame, append 0 or 1 to input_motion_buffer
		"""
		division = float(numpy.sum(numpy.bitwise_and(frame, ROI))) / numpy.sum(ROI)
		#print('input division', division)
		self.input_motion_buffer.append(True if division >= self.motion_threshold else False)

	def binarize_groundtruth(self, frame):
		"""Convert the groundtruth to binary values
		"""
		# background: 0-25 (around 0) => 0
		# hard shadow: 26-67 (around 50) => 255
		# outside ROI: 68-127 (around 85) => 0
		# unknown motion: 128-212 (around 170) => 255
		# motion: 213-255 (around 255) => 255
		binarized_groundtruth = frame
		binarized_groundtruth.setflags(write=1)
		binarized_groundtruth[numpy.bitwise_and(frame>=0,frame<=25)]=0
		binarized_groundtruth[numpy.bitwise_and(frame>=26,frame<=67)]=255
		binarized_groundtruth[numpy.bitwise_and(frame>=68,frame<=127)]=0
		binarized_groundtruth[numpy.bitwise_and(frame>=128,frame<=212)]=255
		binarized_groundtruth[numpy.bitwise_and(frame>=213,frame<=255)]=255

		return binarized_groundtruth.astype(numpy.uint8)

	def groundtruth_motion(self, frame):
		"""For each groundtruth frame, append 0 or 1 to groundtruth_motion_buffer
		"""
		division = float(numpy.sum(numpy.bitwise_and(frame, ROI))) / numpy.sum(ROI)
		#print('groundtruth division', division)
		self.groundtruth_motion_buffer.append(True if division >= self.motion_threshold else False)

	def precision(self):
		"""Calculate the precision between input_motion_buffer and groundtruth_motion_buffer
		"""
		# precision = (# True positive) / (# Test outcome positive)
		TP = numpy.sum(numpy.logical_and(self.input_motion_buffer, self.groundtruth_motion_buffer))
		#print('TP: ', TP)
		pP = numpy.sum(self.input_motion_buffer)
		#print('predicted P', pP)
		if pP != 0:
			precision = float(TP) / pP
		else:
			precision = 0
		print('precision: ', precision)
		return precision

	def recall(self):
		"""Calculate the recall between input_motion_buffer and groundtruth_motion_buffer
		"""
		# recall = (# True positive) / (# Condition outcome positive)
		TP = numpy.sum(numpy.logical_and(self.input_motion_buffer, self.groundtruth_motion_buffer))
		#print('TP: ', TP)
		#inverted_input = numpy.where(self.input_motion_buffer == True, False, True)
		#FN = numpy.sum(numpy.logical_and(inverted_input, self.groundtruth_motion_buffer))
		aP = numpy.sum(self.groundtruth_motion_buffer)
		#print('actual P: ', aP)
		if aP != 0:
			recall = float(TP) / aP
		else:
			recall = 0
		print('recall: ', recall)
		return recall

class MeanMotionEstimator(Estimator):
	"""This estimator calculates the background by taking the mean of the X previous frames
	"""
	def feed(self, frame):
		"""Feed the motion detector with a frame and return a binary foreground mask
		"""
		self.frame_count += 1
		print(self.frame_count)
		if self.frame_count > self._max_buffer_size:
			#cv2.imshow('Video', numpy.bitwise_and(frame, ROI))
			# compute background model

			background_model = numpy.mean(numpy.array(self._buffer), axis=0).astype(numpy.uint8)
			# write background model with ScreenWriter
			#cv2.imshow('Background Model', numpy.bitwise_and(background_model, ROI))
			# cast to signed integers
			frame = frame.astype(int)
			background_model = background_model.astype(int)
			# substract frame from background model
			foregroundmask = numpy.absolute(background_model - frame).astype(numpy.uint8)
			# if absolute values > foreground_mask_threshold 1, else 0 => binarized foregroundmask
			foregroundmask = numpy.where(foregroundmask > self.foreground_mask_threshold, 255, 0).astype(numpy.uint8)
			# write foreground mask with ScreenWriter
			cv2.imshow('Foreground Mask', numpy.bitwise_and(foregroundmask, ROI))
			cv2.waitKey(1)
			# return foregroundmask
			return foregroundmask
		return None

class BlockBasedEstimator(Estimator):
	"""This estimator detects motion using a threshold on the motion vector size of the different blocks
	"""
	# tThe block size
	block_size = 8

	# The motion vector size threshold: 2 for search 5, 4 for search 13
	motion_vector_size_threshold = 2

	# Vary search range between 5 and 13 (respectively 25 and 169 evaluations per block) in steps of 2
	search_range = 5


	# backdoor: 1450; bungalow: 344
	def feed(self, frame):
		self.frame_count+=1
		if(self.frame_count%100 == 0):
			print(self.frame_count)
		#if(self.frame_count > 344):
		if self.frame_count > self._max_buffer_size:
			prev_frame = self._buffer[0]

			# Variables
			rows = len(frame)
			cols = len(frame[0])
			shift = self.search_range//2
			# Initialiseer masker volledig zwart
			foregroundmask = numpy.zeros((rows, cols), dtype=numpy.uint8)
			
			# Itereer over linkerbovenhoeken van alle blokken (0 tot lengte frame, in stappen van block_size)
			for i in range(0,rows, self.block_size):
				for j in range(0,cols, self.block_size):

					# Einde van de blokken: voor randgevallen
					i_upper = numpy.minimum(i+self.block_size, rows-1)
					j_upper = numpy.minimum(j+self.block_size, cols-1)

					# Pak het huidige blok
					cur_block = frame[i:i_upper,j:j_upper]

					# Variablen: normaal 8x8, maar anders vor grenzen
					block_rows = len(cur_block)
					block_cols = len(cur_block[0])

					# Als huidig en vorig exact overeen komen, skip
					prev_block_same = prev_frame[i:i_upper,j:j_upper]
					if(not(numpy.array_equal(cur_block,prev_block_same))):

						# Variabelen voor minimum
						minimum = numpy.Inf
						offsetI = 0
						offsetJ = 0

						# Voor elke offset
						for offRow in range(numpy.maximum(0,i-shift), numpy.minimum(i+shift, rows-block_rows)):
							for offCol in range(numpy.maximum(0,j-shift), numpy.minimum(j+shift, cols-block_cols)):

									prev_block = prev_frame[offRow:(offRow+block_rows),offCol:(offCol+block_cols)]

									# MSE van huidig blok en vorig offset blok
									MSE_no_denom = numpy.sum(numpy.subtract(cur_block, prev_block)**2)
									#if(self.frame_count > 344 and self.frame_count < 350):
										#print(MSE_no_denom)
									# Zoek kleinste MSE en remember offset
									if(MSE_no_denom < minimum):
										minimum = MSE_no_denom
										offsetI = offRow-i
										offsetJ = offCol-j

						#print("MIN: ", float(minimum), " - ", minimum == 0, " at ", offsetI, "-", offsetJ)
						# Zoek Euclidische afstand tussen linkerbovenhoeken
						dist = numpy.sqrt(offsetI**2+offsetJ**2)

						# Groter dan treshold => kleur huidig blok wit
						if(dist>self.motion_vector_size_threshold):
							#print("		WHITENING: ", i, "-", (i+block_rows), " x", j, "-", (j+block_cols))
							foregroundmask[i:(i+block_rows),j:(j+block_cols)] = 255

			# write foreground mask with ScreenWriter
			cv2.imshow('Foreground Mask', numpy.bitwise_and(foregroundmask, ROI))
			cv2.waitKey(1)


			# return foregroundmask
			return foregroundmask

		return None

class MOGOpenCVEstimator(Estimator):
	"""This estimator uses a Mixture of Gaussians (MoG)
	"""
	def feed(self, frame):
		self.frame_count += 1
		if (self.frame_count%100 == 0):
			print(self.frame_count)

		foregroundmask = self.mog.apply(frame)

		if self.frame_count > self._max_buffer_size:
			# write foreground mask with ScreenWriter
			cv2.imshow('Foreground Mask', numpy.bitwise_and(foregroundmask, ROI))
			cv2.waitKey(1)
			return foregroundmask

		return None


# main
if __name__ == "__main__":
	Gst.init(None)

	# parameters
	# X, vary between 5 and 50
	X = 120
	# Foreground mask threshold, fixed on 51
	foreground_mask_threshold = 51
	path = '/home/dma/Downloads/DMA_3_video/baseline/backdoor/'

	# readers
	roi_reader = ROIReader()
	input_reader = InputReader()
	groundtruth_reader = GroundtruthReader()

	# read in ROI once
	roi_reader.read(path + 'ROI.bmp')
	# read in input frame by frame
	input_video = input_reader.read(path + 'input.mkv')
	# read in groundtruth frame by frame
	groundtruth_video = groundtruth_reader.read(path + 'groundtruth.mp4')

	# estimator
	#estimator = MeanMotionEstimator(X, foreground_mask_threshold)
	#estimator = BlockBasedEstimator(1,foreground_mask_threshold)
	estimator = MOGOpenCVEstimator(X, foreground_mask_threshold)

	# main loop
	for (input_frame, groundtruth_frame) in zip(input_video, groundtruth_video):
		estimator.estimate(input_frame, groundtruth_frame)
	
	estimator.precision()
	estimator.recall()
