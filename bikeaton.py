import numpy as np
import cv2
from sklearn.neighbors import KDTree
from itertools import permutations
import random 

def triangle_area(p):
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    return abs((p1[0]*(p2[1]-p3[1]) + 
                p2[0]*(p3[1]-p1[1]) +
                p3[0]*(p1[1]-p2[1]))/2)
    
def strightness(a,b,y):
    x = 0
    if ((a[0][0]-y[0])**2+(a[0][1]-y[1])**2) <= ((b[0][0]-y[0])**2+(b[0][1]-y[1])**2):
        x += 1
    if ((a[1][0]-y[0])**2+(a[1][1]-y[1])**2) <= ((b[1][0]-y[0])**2+(b[1][1]-y[1])**2):
        x += 1
    if ((a[2][0]-y[0])**2+(a[2][1]-y[1])**2) <= ((b[2][0]-y[0])**2+(b[2][1]-y[1])**2):
        x += 1
    return x >= 2
    
    
def dist2p(x):
    p1,p2,p3 = x
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[0]-p3[0]) + abs(p1[1]-p3[1])

def measure_green(img,p1,p2):
    #mesures the amount of green colour around two points
    # returns 0 if it's really green 255 if it's not green at all
    greenlow = 36
    greenup = 86
    ret = 0
    x1 = max(p1[0]-10,0)
    x2 = min(p1[0]+10,img.shape[0])
    y1 = max(p1[1]-10,0)
    y2 = min(p1[1]+10,img.shape[0])
    colour = 0
    for i in range(x1,x2):
        for j in range(y1,y2):
            colour += img[i,j,0]
    colour = colour/200
    if colour < greenup and colour > greenlow:
        ret += 0
    else:
        ret += min(abs(colour-greenup),abs(colour-greenlow))
        x1 = max(p1[0]-10,0)
    x2 = min(p2[0]+10,img.shape[0])
    y1 = max(p2[1]-10,0)
    y2 = min(p2[1]+10,img.shape[0])
    colour = 0
    for i in range(x1,x2):
        for j in range(y1,y2):
            colour += img[i,j,0]
    colour = colour/200
    if colour < greenup and colour > greenlow:
        ret += 0
    else:
        ret += min(abs(colour-greenup),abs(colour-greenlow))
        
    return min(ret,255)
#the photo we will attempt to measure from
img = cv2.imread("roadway.png")
img = cv2.bilateralFilter(img,15,150,150)



#both of these png are close up of roads, need to get the colour to mask with
colours = cv2.imread("ApplicationFrameHost_2019-05-18_15-39-45.png")
colours = cv2.cvtColor(colours,cv2.COLOR_BGR2HSV)
first = []
nd = []
rid = []
for i,x in enumerate(colours):
    for j,y in enumerate(x):
        first.append(y[0])
        nd.append(y[1])
        rid.append(y[2])
blue_lower=np.array([min(first),min(nd),min(rid)],np.uint8)
blue_upper=np.array([max(first),max(nd),max(rid)],np.uint8)
colours = cv2.imread("ApplicationFrameHost_hSGq8kb1Ux.png")
colours = cv2.cvtColor(colours,cv2.COLOR_BGR2HSV)
first = []
nd = []
rid = []
for i,x in enumerate(colours):
    for j,y in enumerate(x):
        first.append(y[0])
        nd.append(y[1])
        rid.append(y[2])
blue_lower=np.array([min(min(first),blue_lower[0]),min(min(nd),blue_lower[1]),
                     min(min(rid),blue_lower[2])],np.uint8)
blue_upper=np.array([max(max(first),blue_upper[0]),max(max(nd),blue_upper[1]),
                     max(max(rid),blue_upper[2])],np.uint8)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, blue_lower, blue_upper)

#clean up the mask a little
mask = cv2.erode(mask, None, iterations=1)
mask = cv2.dilate(mask, None, iterations=1)
mask = cv2.dilate(mask, None, iterations=1)
mask = cv2.erode(mask, None, iterations=1)

edges = cv2.Canny(mask,700,1200)




contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
copy = []
# only want road edges i.e big contours
for x in contours:
    if len(x) > 200:
        xx = max([y[0][0] for y in x])
        xy = min([y[0][0] for y in x])
        yy = max([y[0][1] for y in x])
        yx = min([y[0][1] for y in x])
        if(max(abs(xx-xy),abs(yy-yx))) > 250:
            copy.append(x)
        
contours = copy

copy = mask.copy()
copy.fill(0)
cv2.fillPoly(copy, pts = contours, color=(255,255,255))
#copy = cv2.cvtColor(copy,cv2.COLOR_GRAY2BGR)


copy = cv2.dilate(copy, None, iterations=40)
copy = cv2.erode(copy, None, iterations=40)

contours, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


pairs = []
areas = []
results = []
#contours = list(contours)
size = (0,2)
for x in contours:
    x = size[0] + x.shape[0]
    size = (x,2)
x = np.zeros(size, dtype =int)
yeet = 0
for y in contours:
    x[yeet:y.shape[0]+yeet,:] = y.reshape(y.shape[0],2)
    yeet += y.shape[0]
contours = x
tree = KDTree(contours) 
#build a KD tree of road edges

for j in range(len(contours)):
    print(str(j) + "/" + str(len(contours)))
    y = contours[j]
#        print(y)
#        print(len(x))
    results = []
    ind, dist = tree.query_radius(y.reshape(1,-1), 90, return_distance=True)
    #search the tree for the closes points
    #note searches in a circle so need to calculate which side of the road the points are on somehow
    if not dist.size == 0:
        dist = dist[0].tolist()
        ind = ind[0].tolist()
        for l in range(len(dist)):
            if dist[l] > 25:
                results.append((dist[l],contours[ind[l]]))
    results = sorted(results,key=lambda x:x[0],reverse = True)
    results = [x[1] for x in results]
    #need to find points that maxmise the area of a equalteral trinagle (ie one point on the other side of the road and two point on the same side of the road)
    if len(results) > 3:
        p1 = results[0]
        p2 = results[1]
        p3 = results[2]
        results = results[3:]
        for j in range(len(results)):
            test = results[j]
            perms = list(permutations([test,p1,p2])) + list(permutations([test,p1,p3])) + list(permutations([test,p3,p2]))
            new_area = []
            area2coord = {}
            for p in perms:
                new_area.append(triangle_area(p))
                area2coord[triangle_area(p)] = p
            if max(new_area) > 2500 and max(new_area) >  (0.95 * triangle_area([p1,p2,p3])) and strightness(area2coord[max(new_area)], [p1,p2,p3], y):
                p1 = area2coord[max(new_area)][0]
                p2 = area2coord[max(new_area)][1]
                p3 = area2coord[max(new_area)][2]
        perms = list(permutations([p1,p1,p2]))
        new_area = []
        area2coord = {}
        #take the point that miminses point to point distence between the three points as this will be the one on the other side of the road
        for p in perms:
            new_area.append(dist2p(p))
            area2coord[dist2p(p)] = p
        pairs.append((y,area2coord[min(new_area)][0]))

            
            
        
    

copy = np.full(img.shape, 0)
hue = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
for x in pairs:
    if abs(x[0][0]-x[0][1])+abs(x[1][0]-x[1][1]) > 450:
        cv2.line(copy,(x[0][0],x[0][1]),(x[1][0],x[1][1]),(measure_green(hue,x[0],x[1]),255,0),2)

x = np.nonzero(copy)
img[x] = copy[x]

cv2.startWindowThread()
cv2.namedWindow("contours")
cv2.imshow('contours', img)
cv2.imwrite( "yeet5.jpg", img)
cv2.waitKey(0)




