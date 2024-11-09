import cv2
import numpy as np

subject=cv2.imread("/home/satvik/Pictures/subject.jpeg")
target=cv2.imread("/home/satvik/Pictures/target.jpeg")

subject_gs=cv2.cvtColor(subject,cv2.COLOR_BGR2GRAY)
target_gs=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)

sift=cv2.SIFT_create()
kp_sub,descp_sub=sift.detectAndCompute(subject_gs,None)
kp_targ,descp_targ=sift.detectAndCompute(target_gs,None)

btf=cv2.BFMatcher()
matches=btf.knnMatch(descp_sub,descp_targ,k=2)

good_matches=[]

for m1,m2 in matches:
    if m1.distance<0.64*m2.distance:
        good_matches.append([m1])

fin_img=cv2.drawMatchesKnn(subject,kp_sub,target,kp_targ,good_matches,None,flags=2)

cv2.imshow("Brute Force Object Detection with SIFT Descriptors",fin_img)

cv2.waitKey(10000)
    
