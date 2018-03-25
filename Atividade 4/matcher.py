
def match(img1,img2):

	sift = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	good = []
	for m,n in matches:
		if m.distance < 0.3*n.distance:
			good.append(m)
		
	return good