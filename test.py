import sys
import math

from PIL import Image

'''inim = Image.open("images/0.png")

h = round(inim.size[1] / 2)
w = 3 * h

outim = Image.new("RGB", (w, h * 2), "black")

for x in range(0, w):
	for y in range(0, h):
		r = y
		p = 2 * math.pi * x / w
		ix = round(h - r * math.cos(p))
		iy = round(h + r * math.sin(p))
		# print(ix, iy)
		try:
			outim.putpixel((x, y), inim.getpixel((ix, iy)))
		except Exception as e:
			print(e, (ix, iy))'''

#outim.save("result.png")


import cv2

cap = cv2.VideoCapture(0)

count = 10

while True:
	ret, frame = cap.read()

	if not ret:
		break

	if cv2.waitKey(1) & 0xff == ord('q'):
		cv2.imwrite("calib/" + str(count) + ".jpg", frame)
		count += 1

	cv2.imshow('frame', frame)
