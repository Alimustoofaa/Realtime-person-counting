import numpy as np
import os, sys, argparse
import cv2, imutils, dlib
import time, schedule, datetime
from imutils.video import VideoStream, FPS

# library pysearchimage
from lib import config, thread
from lib.logCountPerson import logPerson
from lib.centroidTracker import CentroidTracker
from lib.trackableobject import TrackableObject

# Path models and prototxt MobileNet SSD
PATH_PROTOTXT   = os.path.join('models/MobileNetSSD_deploy.prototxt')
PATH_MODEL      = os.path.join('models/MobileNetSSD_deploy.caffemodel')

if not os.path.isfile(PATH_MODEL) or not os.path.isfile(PATH_PROTOTXT):
	sys.exit('Path models and prototxt MobileNet SSD not exits')

NET = cv2.dnn.readNetFromCaffe(PATH_PROTOTXT, PATH_MODEL)

# Initialize the list of class label MobileNet SSD
CLASSES = [
	'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
	'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'hourse',
	'motorbike', 'person', 'porredplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Initialize t0 while app run
t0 = time.time()

def main():
	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', type=str,
		help='path to optional input video file')
	parser.add_argument('-o', '--output', type=str,
		help='path to optional output video file')
	
	parser.add_argument('-c', '--confidence', type=float, default=0.4,
		help='minumum probbility to filter weak detections')
	parser.add_argument('-s', '--skip-frames', type=int, default=30,
		help='skip frames between detection')
	args = vars(parser.parse_args())

	if not args.get('input', False):
		print('[INFO] Starting the live stream..')
		vs = VideoStream(config.url).start()
		time.sleep(2.0)
	else:
		if args.get('input') == 'camera':
			print('[INFO] Starting the live from camera..')
			vs = cv2.VideoCapture(0)
			time.sleep(2.0)
		else:
			if not os.path.isfile(args['input']):
				sys.exit('Input video name '+args['input']+' doesnt exit')
			print('[INFO] Starting the video from path..')
			vs = cv2.VideoCapture(args['input'])
	
	writer 	= None
	W 		= None
	H 		= None

	# Centroid tracker
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackableObjects    = {}
	trackers            = []
	x                   = []
	empty               = []
	empty1              = []
	totalFrames         = 0
	totalDown           = 0
	totalUp             = 0
	fps                 = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)
	
	# loop frames from video sream
	while True:
		frame = vs.read()
		frame = frame[1] if args.get('input', False) else frame
		if args['input'] is not None and frame is None:
			break
		frame = imutils.resize(frame, width=500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		if W is None or H is None:
			H, W = frame.shape[:2]
		
		# writing video save to disk
		if args['output'] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			writer = cv2.VideoWriter(args['output'], fourcc, 30, (W, H), True)
		
		# initialize the current status from person detection
		status = 'Waiting'
		rects  = []

		if totalFrames % args['skip_frames'] == 0:
			status = 'Detecting'
			trackers = []

			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			NET.setInput(blob)
			detections = NET.forward()
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > args['confidence']:
					idx = int(detections[0, 0, i, 1])
					if CLASSES[idx] != 'person':
						continue
					# compute the (x, y)-coordinates of the bounding box
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					startX, startY, endX, endY = box.astype('int')
					# construct a dlib rectangle object from the bounding
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)
					trackers.append(tracker)
		else:
			for tracker in trackers:
				status = 'Tracking'
				tracker.update(rgb)
				pos = tracker.get_position()

				staritX  = int(pos.left())
				startY  = int(pos.top())
				endX    = int(pos.right())
				endY    = int(pos.bottom())
				rects.append((startX, startY, endX, endY))

		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, '-Prediction border - Entrance-', 
					(10, H - ((i * 20) + 200)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
		
		objects = ct.update(rects)
		for objectID, centroid in objects.items():
			to = trackableObjects.get(objectID, None)
			if to is None:
				to = TrackableObject(objectID, centroid)
			else:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)
				if not to.counted:
					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						empty.append(totalUp)
						to.counted = True
					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						empty1.append(totalDown)
						x = []
						x.append(len(empty1)-len(empty))

						# if the people limit exceeds over threshold, send an email alert
						if sum(x) >= 80:
							cv2.putText(frame, '-ALERT: People limit exceeded-', 
										(10, frame.shape[0] - 80),
										cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							if config.ALERT:
								print('[INFO] Sending email alert..')
								Mailer().send(config.MAIL)
								print('[INFO] Alert sent')

						to.counted = True
			trackableObjects[objectID] = to
			
			text = 'Id {}'.format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# Construct a tuple of information we will be displaying on the
		info = [('Exit', totalUp),
				('Enter', totalDown),
				('Status', status)]
		info2 = [('Total people inside', x)]

		print('Count Exit : ', totalUp)
		print('Count Enter : ', totalDown)
		print('------------------------')

		# Display output
		for i, (k, v) in enumerate(info):
			text = '{}: {}'.format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
		for i, (k, v) in enumerate(info2):
			text = '{}: {}'.format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# save to csv file
		if config.Log:
			try:
				logPerson(x, empty, empty1)
			except PermissionError as error:
				print('[ERROR] '+error.strerror)

		cv2.imshow('Real-Time Monitoring', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		totalFrames += 1
		fps.update()

		# Time to stop live stream
		if config.Timer:
			t1 = time.time()
			count = t1 - t0
			if count > (config.Minutes * 60):
				break
	fps.stop()
	print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
	print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
	cv2.destroyAllWindows()
	
if config.Scheduler:
	schedule.every().day.at('16:18').do(main)
	while True:
		schedule.run_pending()
else:
	main()