import cv2
import numpy as np


Image = 'Circle.jpg'					#Bild Datei
WindowWidth = 1100					#Fenster Breite
lower_Color_val = (0, 0, 0) 		#Untere RGB Farbgrenze
higher_Color_val = (100, 100, 100)	#Obere RGB Farbgrenze


#Dient zur Anpassung der Fenstergröße
def resize(Images, ImagesWidth):
	ImagesLen = len(Images)
	height = Images[0].shape[0]
	width = Images[0].shape[1] * ImagesLen
	try:
		widthDim = int(Images[0].shape[1] - (width - ImagesWidth) / ImagesLen) 	#Die nötige Länge eines Bildes wird ausgerechnet
		heightDim = int(height / ((width / ImagesWidth)))						#Die Höhe wird ausgerechnet
	except:
		widthDim = heightDim = 1
	ResizedImages = []
	for Image in Images:
		ResizedImages.append(cv2.resize(Image, (widthDim, heightDim), interpolation = cv2.INTER_AREA))
	return ResizedImages

def DetectObject(mask, InputImage):
 	Contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	#Konturen werden gesucht
 	OutputImage = InputImage
 	centerX = None
 	centerY = None
 	for cnt in Contours:
 		approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)		#Aproximation des Objekts
 		M = cv2.moments(cnt)	#In späterer Folge zur ermittlung des Mittelpunkts erforderlich
 		if M["m00"] != 0:
 			if 13 <= len(approx) <= 19 or len(approx) == 4:

 				OutputImage = cv2.drawContours(InputImage, [cnt], -1, (0,255,0), -1)
		 		
		 		#Mittelpunkte werden ermittelt
		 		centerY = int(M["m01"] / M["m00"]) 
			 	centerX = int(M["m10"] / M["m00"])

			 	#Einzeichnen des Mittlelpunkts
			 	cv2.circle(OutputImage, (centerX, centerY),10,(0,0,255), -1)	

	 			#Wenn ein Rechteck erkannt wird
			 	if len(approx) == 4:
			 		cv2.putText(OutputImage, "Rechteck",(centerX, centerY),cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

	 			# Sonst Kreis wird erkannt
		 		else:
			 		cv2.putText(OutputImage, "Kreis",(centerX, centerY),cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
			 	
 	return OutputImage, centerX, centerY


IMG = cv2.imread(Image)

blur = cv2.GaussianBlur(IMG,(13,13),0) #Verringerung von Bild rauschen

RGB = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

mask = cv2.inRange(RGB, lower_Color_val, higher_Color_val) #Grenzwerte

OutputImage, centerX, centerY = DetectObject(mask, RGB)

print("Mittelpunkt:\n""X: "+ str(centerX) + " pixel" + "\nY: " + str(centerY) + " pixel") #Der Abstand vom Kreis Objekt Mittelpunkt relativ zur oberen linken Ecke wird ausgegeben

thresholdblur = cv2.bitwise_and(blur, blur, mask=mask)	#Das Bild mit den eingestellten Grenzwerten wird erstellt

blur, thresholdblur, OutputImage = resize((blur, thresholdblur, OutputImage), WindowWidth)

CombinedImages = np.concatenate((blur, thresholdblur, OutputImage), axis=1) #Bilder werden kombiniert
	
cv2.imshow('Rechteck oder Kreis', CombinedImages)
	    
cv2.waitKey(0)
cv2.destroyAllWindows()