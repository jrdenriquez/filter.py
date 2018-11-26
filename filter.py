from PIL import Image, ImageDraw
import imutils
import numpy as np
import face_recognition
import cv2
import time
face_locations = []
THRESHX = 600
THRESHY = 400
class Roi():
	def __init__(self,y,h,x,w):
		self.y = y*4
		self.h = h*4
		self.x = x*4
		self.w = w*4
	
	def face_points(self):
		face_x1 = self.x 
		face_y1 = self.y + 20
		face_x2 = self.w + 5	
		face_y2 = self.h + 60

		face_x2 = THRESHX-1 if face_x2 >= THRESHX else face_x2
		face_y2 = THRESHY-1 if face_y2 >= THRESHY else face_y2

		return face_y1,face_y2,face_x1,face_x2
	def hair_points(self):
		hair_x = self.x - 35
		hair_y = self.y - 96
		hair_w = self.w + 35
		hair_h = self.h - 110
		
		hair_y = 0 if hair_y < 0 else hair_y
		hair_x = 0 if hair_x < 0 else hair_x
		hair_w = THRESHX-1 if hair_w >= THRESHX else hair_w
		hair_h = THRESHY-1 if hair_h >= THRESHY*0.8 else hair_h
		return hair_y,hair_h,hair_x,hair_w

class Images():
	def __init__(self,image,dim):
		self.image = image
		self.dim = dim
	def resize(self):
		self.image = cv2.resize(self.image, self.dim, interpolation = cv2.INTER_AREA)
		return self.image
	def masking(self,src,masked):
		self.mask = cv2.bitwise_and(src,src,mask=masked)
		return self.mask
	def combine(self,src1,src2):
		self.sum = cv2.add(src1,src2)
		return self.sum

face_cover = cv2.imread('mask2.png',-1)
hair_cover= cv2.imread('hair.png',-1)

# Create the mask for the images
masked_face = face_cover[:,:,3]
masked_hair = hair_cover[:,:,3]

# Create the inverse of mask for the images
masked_face_inv = cv2.bitwise_not(masked_face)
masked_hair_inv = cv2.bitwise_not(masked_hair)

# Convert mustache image to BGR
hair_cover = hair_cover[:,:,0:3]
face_cover = face_cover[:,:,0:3]

print("[INFO] camera sensor warming up...")
video_capture = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    # Grab a single frame of video
	ret, frame = video_capture.read()
	frame = imutils.resize(frame, width=600,height=400)

    # Resize frame of video to 1/4 size for faster face detection processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
	gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
	face_locations = face_recognition.face_locations(gray, model="cnn")
	face_list = face_recognition.face_landmarks(frame)
	
	
	for (fy, fw, fh, fx) in face_locations:						
				
		#face = cv2.rectangle(frame,(fx,fy),(fw,fh),(255,255,0),3)
		#cv2.imshow('video', frame)
		face_filter = Roi(fy,fh,fx,fw)
		points = face_filter.face_points()
		roi_face = frame[points[0]:points[1],points[2]:points[3]]				

		h,w = (points[1]-points[0]),(points[3]-points[2])
		dim = (w,h)
		
		face_cover = Images(face_cover, dim).resize()
		masked_face = Images(masked_face, dim).resize()
		masked_face_inv = Images(masked_face_inv, dim).resize()
					
		roi_bg = Images(roi_face,dim).masking(roi_face, masked_face_inv)
		roi_fg = Images(face_cover,dim).masking(face_cover, masked_face)
		
		dst = Images(roi_bg,dim).combine(roi_bg,roi_fg)
        
		frame[points[0]:points[1],points[2]:points[3]] = dst
		#cv2.imshow('Mask test', frame)
		
#-------------------------------------------------------			
		hair_filter = Roi(fy,fh,fx,fw)
		points = hair_filter.hair_points()
		roi_hair = frame[points[0]:points[1],points[2]:points[3]]	
			
		h2,w2 = (points[1]-points[0],points[3]-points[2])
		dim2 = (w2,h2)
				
		hair_cover = Images(hair_cover, dim2).resize()
		masked_hair = Images(masked_hair, dim2).resize()
		masked_hair_inv = Images(masked_hair_inv, dim2).resize()
		
		roi_bg2 = Images(roi_hair,dim2).masking(roi_hair, masked_hair_inv)
		roi_fg2 = Images(hair_cover,dim2).masking(hair_cover,  masked_hair)

		dst2 = Images(roi_bg,dim2).combine(roi_bg2,roi_fg2)
		frame[points[0]:points[1],points[2]:points[3]] = dst2
		cv2.imshow('Hair cover',dst2)	
# ----------------------------------------------------	
	for face_landmarks in face_list:
		canvas = Image.fromarray(frame)
		d = ImageDraw.Draw(canvas, 'RGBA')
		d.polygon(face_landmarks['left_eyebrow'], fill=(255, 255, 255, 240))
		d.polygon(face_landmarks['right_eyebrow'], fill=(255, 255, 255, 240))
			
		d.polygon(face_landmarks['left_eye'][0:2]+face_landmarks['left_eye'][5:6], fill=(0, 0, 0, 255))
		d.polygon(face_landmarks['right_eye'][0:2]+face_landmarks['right_eye'][5:6], fill=(0, 0, 0, 255))
			
		d.polygon(face_landmarks['left_eye'][2:5], fill=(0, 0, 0, 255))
		d.polygon(face_landmarks['right_eye'][2:5], fill=(0, 0, 0, 255))
		d.ellipse(face_landmarks['left_eye'][1]+face_landmarks['left_eye'][4], fill=(0,0, 255, 100))
		d.ellipse(face_landmarks['right_eye'][1]+face_landmarks['right_eye'][4], fill=(0, 0, 255, 100))
			
		canvas = np.array(canvas)
		frame = canvas
		break
	
	cv2.imshow('Video', frame)
		
    # Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
