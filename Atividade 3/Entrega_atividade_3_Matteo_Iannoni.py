import cv2
import numpy as np

###################################################
# O USO DE UM FILTRO PROS 'GOOD MATCHES' FAZ COM QUE HAJA UM DELAY DE ADAPTACAO
# (ao omitir a raposa, o programa ainda vai printar 'raposa encontrada' por alguns segundos)
# ISSO NAO É UM ERRO, É UMA CARACTERISTICA DE USAR O FILTRO PARA RESULTADOS MAIS
# CONSISTENTES AO ACHAR A RAPOSA E AO NAO ACHAR, COM ISSO VOCE PODE CHACOALHAR A
# RAPOSA E O PROGRAMA AINDA RECONHECE ELA.
###################################################


cap = cv2.VideoCapture(0)

LAST_GOODS = []

img1 = cv2.cvtColor(cv2.imread("madfox.jpg"), cv2.COLOR_BGR2GRAY)

def auto_canny(image, sigma=0.05):
	v = np.median(image)

	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	return edged

def drawMatches(img1, kp1, img2, kp2, matches):

	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]

	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

	out[:rows1,:cols1] = np.dstack([img1, img1, img1])

	out[:rows2,cols1:] = np.dstack([img2, img2, img2])

	for mat in matches:

		mat = mat[0]

		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

		cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

	cv2.imshow('Matched Features', cv2.resize(out,(0,0),fx=0.5,fy=0.5)) #Essa funcao é usada por curiosidade, pra saber onde os matches estao # Cria um novo cv2.imshow pra mostrar os matches

def simpledrawMatches(img2, kp2, matches):

	out = frame_pure

	for mat in matches:

		mat = mat[0]

		img2_idx = mat.trainIdx

		(x2,y2) = kp2[img2_idx].pt

		cv2.circle(out, (int(x2),int(y2)), 4, (255, 0, 0), 1) # Desenha no proprio frame original os matches (é o que ta em uso)

def findmatches(img2):
	global LAST_GOODS

	kp2, des2 = sift.detectAndCompute(img2,None)

	matches = flann.knnMatch(des1,des2,k=2)

	good = []
	for m,n in matches:
		if m.distance < 0.3*n.distance:
			good.append(m)
	
	# print(np.array(LAST_GOODS).mean()) ## Isso printa o filtro, pra eu saber qual threshold indica a existencia do madfox na imagem
	LAST_GOODS.append(len(good))
	
	if len(LAST_GOODS) > 30: # Aqui eu uso o threshold obtido usando o print
		LAST_GOODS = LAST_GOODS[10:]
	
	if np.array(LAST_GOODS).mean() > 1:
		print('A raposa foi encontrada')

	simpledrawMatches(img2,kp2,matches)

#############
# Eu decidi fazer tudo isso aqui, pra reduzir o processamento ja que isso é necessario só uma vez, posto que a imagem do madfox nao muda
#############

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

############
# Aqui termina
############

while True:

	ret, frame_pure = cap.read()

	compare = cv2.cvtColor(frame_pure, cv2.COLOR_BGR2GRAY)

	frame = auto_canny(frame_pure)

	if frame.any() != None:

		circles = cv2.HoughCircles(frame,cv2.HOUGH_GRADIENT,1.4,70,param1=50,param2=100,minRadius=5,maxRadius=80)

		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

		if circles is not None:

			circles = np.uint16(np.around(circles))

			findmatches(compare)

			for i in circles[0,:]:
				cv2.circle(frame_pure,(i[0],i[1]),i[2],(0,255,0),2) # Eu decidi desenhar os circulos no frame original
				cv2.circle(frame_pure,(i[0],i[1]),2,(0,0,255),3)
			
			#print numero de circulos encontrados
			if len(circles[0,:]) > 1:
				print(str(len(circles[0,:])) + " circles were found")
			else:
				print(str(len(circles[0,:])) + " circle were found")
		else:
			print("No circles were found")

		cv2.imshow('Pure frame com circulos',frame_pure)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()