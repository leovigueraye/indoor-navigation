import streamlit as st
def preprocessing():
    import cv2
    import numpy as np
    from PIL import Image
    import streamlit as st
    import time
    import os

    # Image Preprocessing
    size = 4  # variable for image size

    def delete_files(file_paths):
        for file_path in file_paths:
            try:
                os.remove(file_path)
                # print(f"File {file_path} deleted successfully.")
            except FileNotFoundError:
                print(f"File {file_path} not found.")
            except PermissionError:
                print(f"Permission error. Unable to delete {file_path}.")
            except Exception as e:
                print(f"An error occurred: {e}")

    # Example usage:
    files_to_delete = ["dirtest.png", "maptest.png", "maptestcolor.png"]
    delete_files(files_to_delete)

    def arrayreturn(pts):

        temp_rect = np.zeros((4, 2), dtype="float32")

        s = np.sum(pts, axis=2)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]

        diff = np.diff(pts, axis=-1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

        return temp_rect


    def autoprocess(image):

        k = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
        mapimg = np.array(Image.open(image))
        # np_image = np.array(Image.open(image))
        
        mapgray = cv2.cvtColor(mapimg,cv2.COLOR_BGR2GRAY)
        mapthresh = cv2.adaptiveThreshold(mapgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,7,7)
        
        
        width, height = mapthresh.shape
        
        mapsharp = cv2.filter2D(mapthresh, -1, k)  
        mapinvert = cv2.bitwise_not(mapsharp)  
        mapdilate = cv2.dilate(mapinvert, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations = 3)  
        mapclose = cv2.morphologyEx(mapdilate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)), iterations = 1)
        
        cnts, _ = cv2.findContours(mapclose,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        

        cnts = sorted(cnts, key=cv2.contourArea,reverse=True)
        
        for i in range (0, len(cnts)):
            
            if i != 0:
                continue
            
            x,y,w,h = cv2.boundingRect(cnts[i])
            hull = cv2.convexHull(cnts[i])
            peri = cv2.arcLength(hull,True)
            approx = cv2.approxPolyDP(hull,0.001*peri,True)
            pts = np.float32(approx)  
            
            
            src = arrayreturn(pts)
            dst = np.array([[int(w*0.05),int(h*0.05)],[w+int(w*0.05),int(h*0.05)],[w+int(w*0.05),h-int(h*0.05)],[int(w*0.05), h-int(h*0.05)]], np.float32)
            
            M = cv2.getPerspectiveTransform(src,dst)
            warp = cv2.warpPerspective(mapimg, M, (int(w*1.1),int(h*1.2)))
            
            cv2.imwrite("maptestcolor.png", warp)
            
            warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
            warp = cv2.adaptiveThreshold(warp,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,11)
            
            return warp
    
    def initialize_images(map_image, dir_image):
        processedmap = autoprocess(map_image)
        cv2.imwrite("maptest.png", processedmap)
        st.image(processedmap, caption="Processed Map", use_column_width=True)

        st.text("Map image processed!")
        # confirm_satisfaction(processedmap, "Map Image")

        if dir_image is not None:
            # Save the uploaded file with a new name
            new_name = "dirtest.png"  # You can choose any new name you want
            with open(new_name, "wb") as f:
                f.write(dir_image.read())

            # st.success(f"Image uploaded and renamed to {new_name}")


            # processed_dir = automatic_process_direct(dir_image)
            # st.image(processed_dir, caption="Processed Directory listings", use_column_width=True)
            # confirm_satisfaction_dir(processed_dir, "Directory Image")
        

    # Streamlit app
    st.title("Indoor Navigation System")

    # Sidebar for user inputs
    st.sidebar.header("Image Preprocessing")

    # File uploader for map image
    map_image = st.file_uploader("Upload floor plan Image", type=["jpg", "jpeg", "png"])

    # File uploader for directory image
    dir_image = st.file_uploader("Upload Directory Image", type=["jpg", "jpeg", "png"])

    # Check if images are uploaded
    if map_image is not None and dir_image is not None:
    # Display uploaded images
        st.image([map_image, dir_image], caption=["Map Image", "Directory Image"], width=300)

        # Process and display images
        initialize_images(map_image, dir_image)


def routing():
        
    import streamlit as st
    import io
    import os
    import copy

    # Imports the Google Cloud client library
    from google.cloud import vision

    st.sidebar.header("Routing Process")
    #desktop
    #os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/Marcus/AppData/Local/Google/Cloud SDK/apikey.json"

    #laptop
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/LEONARD EGURIASE/Downloads/awesome-ridge-396020-a6f2b94302c0.json"



    #=============================================================================================================

    # to check if coordinates are inside a polygon
    def inside(x,y,poly):

        n = len(poly)
        inside = False
        
        # p1 variable for first polygon vertice
        p1x,p1y = poly[0]
        
        for i in range(n+1):
            # p2 variable for next polygon vertice 
            p2x,p2y = poly[i % n]
            # if point is within y axis
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    # if point within max of x axis
                    if x <= max(p1x,p2x):
                        # if not same y axis, calculate new x threshold
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        # if x is lesser than new x threshold, point is inside polygon
                        if p1x == p2x or x <= xints:
                            inside = not inside
                            
            # assign 2nd polygon vertice to p1 variable for future comparison with next vertice                
            p1x,p1y = p2x,p2y
        
        # return boolean
        return inside

    #=======================================================================================

    # to get the max y coordinate
    def getYMax(data):
        v = data.text_annotations[0].bounding_poly.vertices
        yArray = []
        for i in range (0,4):
            yArray.append(v[i].y)
        
        return max(yArray)

    #=======================================================================================

    def invertAxis(data, yMax):
        data = fillMissingValues(data)
        for i in range (1, len(data.text_annotations)):
            v = data.text_annotations[i].bounding_poly.vertices
            
            for j in range (0,4):
                v[j].y = yMax - v[j].y
            
        return data

    #=======================================================================================

    def fillMissingValues(data):
        for i in range (1, len(data.text_annotations)):
            v = data.text_annotations[i].bounding_poly.vertices
            
            for vertex in v:
                if vertex.x == None:
                    vertex.x = 0
                
                if vertex.y == None:
                    vertex.y = 0
            
        return data

    #=======================================================================================

    def getBoundingPolygon(mergedArray):

        external = []
        for i in range (0, len(mergedArray)):
            arr = []

            # calculate merged text height (left and right)
            h1 = mergedArray[i].bounding_poly.vertices[0].y - mergedArray[i].bounding_poly.vertices[3].y
            h2 = mergedArray[i].bounding_poly.vertices[1].y - mergedArray[i].bounding_poly.vertices[2].y
            h = h1
            # get larger height value
            if(h2> h1):
                h = h2
            # calculate height threshold for gradient purposes in future
            avgHeight = h * 0.6
            
            # get coordinates for top line of merged text
            arr.append(mergedArray[i].bounding_poly.vertices[1])
            arr.append(mergedArray[i].bounding_poly.vertices[0])
            line1 = getRectangle(copy.deepcopy(arr), True, avgHeight, True)
            
            # get coordinates for bottom line of merged text
            arr = []
            arr.append(mergedArray[i].bounding_poly.vertices[2])
            arr.append(mergedArray[i].bounding_poly.vertices[3])
            line2 = getRectangle(copy.deepcopy(arr), True, avgHeight, False)
            
            # initialize array to store individual merged text bounding box info
            internal = []
            # insert big bounding box coordinates of merged text, index, empty array and matched line boolean
            internal.append(createRectCoordinates(line1, line2))
            internal.append(i)
            internal.append([])
            internal.append(False)
            
            # append individual merged array bounding box info
            external.append(internal)
        
        return external

    #=======================================================================================

    def combineBoundingPolygon(mergedArray, arr):
        # select one merged text from the array
        for i in range (0, len(mergedArray)):
            # get big bounding box coordinates
            bigBB = arr[i][0]

            # iterate through all the array to find the match
            for k in range (i, len(mergedArray)):
                # if its not own bounding box and has never been matched before
                if(k != i and arr[i][3] == False):
                    insideCount = 0
                    
                    # for each coordinate points of merged text
                    for j in range (0,4):
                        coordinate = mergedArray[k].bounding_poly.vertices[j]
                        
                        # check if each coordinate point is inside big bounding box
                        if(inside(coordinate.x, coordinate.y, bigBB)):
                            # increment by 1 if true
                            insideCount += 1
            
                    # if all four point were inside the big bb
                    if(insideCount == 4):
                        # append match info dictionary into array and set matched status as true
                        # matchLineNum indicates the index of matched word in merged text array
                        match = {"matchCount": insideCount, "matchLineNum": k}
                        arr[i][2].append(match)
                        arr[i][3] = True

        return arr

    #=======================================================================================

    def getRectangle(v, isRoundValues, avgHeight, isAdd):
        if(isAdd):
            v[1].y = v[1].y + int(avgHeight)
            v[0].y = v[0].y + int(avgHeight)
        else:
            v[1].y = v[1].y - int(avgHeight)
            v[0].y = v[0].y - int(avgHeight)

        yDiff = (v[1].y - v[0].y)
        xDiff = (v[1].x - v[0].x)
        
        if xDiff != 0: 
            gradient = yDiff / xDiff
        else:
            gradient = 0

        xThreshMin = 1
        xThreshMax = 2000

        if (gradient == 0):
            #extend the line
            yMin = v[0].y
            yMax = v[0].y
        else:
            yMin = (v[0].y) - (gradient * (v[0].x - xThreshMin))
            yMax = (v[0].y) + (gradient * (xThreshMax - v[0].x))
        
        
        if(isRoundValues):
            yMin = int(yMin)
            yMax = int(yMax)
        
        return {"xMin" : xThreshMin, "xMax" : xThreshMax, "yMin": yMin, "yMax": yMax}

    #=======================================================================================

    def createRectCoordinates(line1, line2):
        return [[line1["xMin"], line1["yMin"]], [line1["xMax"], line1["yMax"]], [line2["xMax"], line2["yMax"]],[line2["xMin"], line2["yMin"]]]

    #=============================================================================================================
        
        
    def mergeNearByWords(data):

        yMax = getYMax(data)
        data = invertAxis(data, yMax)
        
        rawText = []

        # Auto identified and merged lines from gcp vision
        lines = data.text_annotations[0].description.split('\n')
        # gcp vision full text
        for data in data.text_annotations:
            rawText.append(data)
        
        #reverse to use lifo
        lines.reverse()
        rawText.reverse()
        #to remove the zeroth element which gives the total summary of the text
        rawText.pop()
        
        # returns array containing all merged nearby texts
        mergedArray = getMergedLines(lines, rawText)
        
        # returns array containing big bounding box info for each merged text
        arr = getBoundingPolygon(mergedArray)

        # returns array that contains words within big bounding boxes
        arr = combineBoundingPolygon(mergedArray, arr)

        #This returns final array containing all lines with shop name and lot number
        finalArray = constructLineWithBoundingPolygon(mergedArray, arr)
        
        return finalArray

    #=======================================================================================

    def constructLineWithBoundingPolygon(mergedArray, arr):
        finalArray = []
        
        # loop through all merged texts
        for i in range (0, len(mergedArray)):
            # only execute on matched words
            if(arr[i][3] == True):
                
                # if there is no matched info
                if(len(arr[i][2]) == 0):
                    finalArray.append(mergedArray[i].description)
                    st.text("xxxxxxxx")
                # append line containing shop name and lot number into array
                else:
                    finalArray.append(arrangeWordsInOrder(mergedArray, i, arr))
        
        # return final array containing all lines with shop name and lot number
        return finalArray

    #=======================================================================================

    def getMergedLines(lines,rawText):

        mergedArray = []
        
        while(len(lines) != 1):
            
            l = lines.pop()
            l1 = copy.deepcopy(l)
            status = True

            while (True):
                wElement = rawText.pop()
                
                if(wElement == None):
                    break;
                
                w = wElement.description

                try:
                    index = l.index(w)
                except ValueError:
                    index = -1
                    continue

                
                #check if the word is inside
                l = l[index + len(w):]
                if(status):
                    status = False
                    #set starting coordinates
                    mergedElement = wElement
                
                if(l == ""):
                    #set ending coordinates
                    mergedElement.description = l1
                    mergedElement.bounding_poly.vertices[1].x = wElement.bounding_poly.vertices[1].x
                    mergedElement.bounding_poly.vertices[1].y = wElement.bounding_poly.vertices[1].y
                    mergedElement.bounding_poly.vertices[2].x = wElement.bounding_poly.vertices[2].x
                    mergedElement.bounding_poly.vertices[2].y = wElement.bounding_poly.vertices[2].y
                    mergedArray.append(mergedElement)
                    
                    break

        return mergedArray

    #=======================================================================================

    def arrangeWordsInOrder(mergedArray, k, arr):
        mergedLine = ''
        
        # get matched info of selected merged text
        line = arr[k][2]
        
        for i in range (0, len(line)):
            # get index of matched word
            index = line[i]["matchLineNum"]
            # get text info of matched word
            matchedWordForLine = mergedArray[index].description

            mainX = mergedArray[k].bounding_poly.vertices[0].x
            compareX = mergedArray[index].bounding_poly.vertices[0].x
            
            # if matched word is positioned after word in k position of mergeeArray
            if(compareX > mainX):
                mergedLine = mergedArray[k].description +':'+ matchedWordForLine
            else:
                mergedLine = matchedWordForLine + ':' + mergedArray[k].description
        
        # return final line that contains shop name and lot number
        return mergedLine


    #=============================================================================================================

    def receiveimg_directory_text(filename):
        # Instantiates a client
        client = vision.ImageAnnotatorClient()
        # The name of the image file to annotate
        file_name = os.path.join(
            os.path.dirname(__file__),
            filename)
        
        # Loads the image into memory
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        response = client.text_detection(image=image)
        
        result = mergeNearByWords(response)
        
        return result

    #=======================================================================================

    def receiveimg_map_text(filename):
        # Instantiates a client
        client = vision.ImageAnnotatorClient()
        # The name of the image file to annotate
        file_name = os.path.join(
            os.path.dirname(__file__),
            filename)
        
        # Loads the image into memory
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        response = client.text_detection(image=image)
        
        texts = response.text_annotations
        
        arr = []
        for text in texts:
            arr.append(text)
        
        return arr

    #=======================================================================================
        
    def return_directorydict():
        # get directory lot and shop name data    
        directory_data = receiveimg_directory_text('dirtest.png')
        
        st.text("info retrieved!")
        
        # initialize dictionary to store lot and label
        directory_dict = {}
        
        # insert directory data into dictionary
        for x in range (len(directory_data)):
            a = directory_data[x]
            b = a.split(':')
            directory_dict[b[0]] = b[1]
            
        return directory_dict

    #=======================================================================================

    def return_maplabel():
        # get map label data and initialize array for the data
        image_path2 = "../maptest.png"
        map_data = receiveimg_map_text(image_path2)
        maplabel = []
        
        # assigning map label into array
        for text in map_data:
            maplabel.append(text.description)
        
        return maplabel

    #=======================================================================================

    import cv2
    from PIL import Image
    import time

    # obtain a dictionary of directory labels and lots
    # directory_dict = vision.return_directorydict()
    directory_dict = return_directorydict()

    # obtain an array of map labels
    # map_data = receiveimg_map_text("maptest.png")
    map_data = receiveimg_map_text("maptest.png")

    st.text("Finished processing map and directory!")
    st.text("number of lots retrieved: " + str(len(directory_dict)))

    size = 4
    count = 0

    #========================================================================== 

    # function to determine pixels that are non-passable (black pixels)
    def is_blocked(p):
        global count
        x,y = p
        pixel = path_pixels[x,y]

        #if (pixel < 255):
        if any(c < 225 for c in pixel):
            return True

    #========================================================================== 
        
    # 4 connectivity neighbours
    def neighbours_4(p):
        x, y = p
        neighbors = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        return [p for p in neighbors if not is_blocked(p)]

    # 8 connectivity neighbours
    def neighbours_8(p):
        x, y = p
        neighbors = [(x-1, y), (x, y-1), (x+1, y), (x, y+1), (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
        return [p for p in neighbors if not is_blocked(p)]

    #========================================================================== 

    # 2 types of heuristics
    def manhattan(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def squared_euclidean(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

    #========================================================================== 

    def return_route(startx, starty, endx, endy):
        global path_pixels
        
        # shrink coordinates 
        start = (startx // size, starty // size)
        goal = (endx // size, endy // size)
        
        # shrink target map image
        
        img_c = cv2.imread("maptestcolor.png")    
        img = cv2.imread("maptest.png",0)
        w,h = img.shape
        w = int(w/size)
        h = int(h/size)
        img = cv2.resize(img, (h,w), interpolation=cv2.INTER_AREA)
        img_c = cv2.resize(img_c, (h,w), interpolation=cv2.INTER_AREA)
        
        # arrays for storing new goal and new starting coordinates
        start_arr = []
        goal_arr = []
        
        # create new coordinates on starting location lot and append into an array
        # right side
        count = 0
        for x in range(start[0],w):
            for y in range(start[1]-1,start[1]):
                channels_xy = img[y,x]
                if (channels_xy < 225):    
                    count += 1
                    if count == 1:
                        start_arr.append((x+5,y))
                    break
        # left side
        count = 0
        for x in range(start[0],0,-1):
            for y in range(start[1]-1,start[1]):
                channels_xy = img[y,x]
                if (channels_xy < 225):    
                    count += 1
                    if count == 1:
                        start_arr.append((x-5,y))
                    break
        # bottom side
        count = 0
        for x in range(start[0]-1,start[0]):
            for y in range(start[1],h):
                channels_xy = img[y,x]
                if (channels_xy < 225):    
                    count += 1
                    if count == 1:
                        start_arr.append((x,y+5))
                    break
        # top side
        count = 0
        for x in range(start[0]-1,start[0]):
            for y in range(start[1],0,-1):
                channels_xy = img[y,x]
                if (channels_xy < 225):    
                    count += 1
                    if count == 1:
                        start_arr.append((x,y-5))
                    break
                    
        # create new coordinates on destination lot and append into an array
        # right side
        count = 0
        for x in range(goal[0],w):
            for y in range(goal[1]-1,goal[1]):
                channels_xy = img[y,x]
                if (channels_xy < 225):    
                    count += 1
                    if count == 1:
                        goal_arr.append((x+5,y))
                    break
        # left side
        count = 0
        for x in range(goal[0],0,-1):
            for y in range(goal[1]-1,goal[1]):
                channels_xy = img[y,x]
                if (channels_xy < 225):    
                    count += 1
                    if count == 1:
                        goal_arr.append((x-5,y))
                    break
        # bottom side
        count = 0
        for x in range(goal[0]-1,goal[0]):
            for y in range(goal[1],h):
                channels_xy = img[y,x]
                if (channels_xy < 225):    
                    count += 1
                    if count == 1:
                        goal_arr.append((x,y+5))
                    break
        # top side
        count = 0
        for x in range(goal[0]-1,goal[0]):
            for y in range(goal[1],0,-1):
                channels_xy = img[y,x]
                if (channels_xy < 225):    
                    count += 1
                    if count == 1:
                        goal_arr.append((x,y-5))
                    break
        
        cv2.imwrite("maptest2.png", img)
        cv2.imwrite("maptestcolor2.png", img_c)
        
        # convert image into RGB
        # load image pixel information
        path_img = Image.open("maptest2.png").convert('RGB')    
        color_img = Image.open("maptestcolor2.png")
        path_pixels = path_img.load()
        path_pixels_c = color_img.load()
        
        # set distance and heuristic types
        distance = manhattan
        heuristic = manhattan
        
        # distance = squared_euclidean
        # heuristic = squared_euclidean
        
        # obtain A Star shortest path
        path = AStar(start_arr, goal_arr, neighbours_8, distance, heuristic)
        
        st.text("plotting route...")
        
        # for position in path:
        #     x,y = position
        #     path_pixels_c[x,y] = (0,0,0) # black 
        
        # increase size of path
        route_thickness = 3
        
        # plot the path in the color red
        for position in path:
            x,y = position
            for i in range(route_thickness):
                path_pixels_c[x + i, y] = (0, 0, 0) # red (255, 5, 5)
        
        
        # save the image with plotted path 
        color_img.save("route.png")
        
    #========================================================================== 

    # A Star Algorithm
    def AStar(start_arr, goal_arr, neighbor_nodes, distance, cost_estimate):
        
        # function to obtain path from start to goal after current node reaches goal 
        def reconstruct_path(came_from, current_node):
            path = []
            
            # while there are still path nodes, append into an array
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]   
            # return the array that contains path
            return list(reversed(path))
        
        start_time = time.time()
        # for each coordinate in starting coordinates array
        for x in range(0,len(start_arr)):
            start = start_arr[x]
            
            # for each coordinate in goal coordinates array
            for y in range (0,len(goal_arr)):
                goal = goal_arr[y]
                
                # For each node, the cost of getting from the start node to that node. (g score)
                # The cost of going from start to start is zero.
                g_score = {start: 0}    
                
                # For each node, the total cost of getting from the start node to the goal
                # by passing by that node. That value is partly known, partly heuristic. (f score)
                # For the first node, that value is completely heuristic.
                f_score = {start: g_score[start] + cost_estimate(start, goal)}
                
                # The set of currently discovered nodes that are not evaluated yet. (openset)
                # Initially, only the start node is known.
                openset = {start}
                
                #The set of nodes already evaluated (closedSet)
                closedset = set()
                
                # For each node, which node it can most efficiently be reached from. (cameFrom)
                # If a node can be reached from many nodes, cameFrom will eventually contain the
                # most efficient previous step.
                came_from = {start: None}
            
                while openset:
                    
                    current = min(openset, key=lambda x: f_score[x])
                    
                    # if goal is reached, return path array
                    if current == goal:
                        st.text("Found route!")
                        st.text("--- %s seconds ---" % (time.time() - start_time))
                        return reconstruct_path(came_from, goal)
                    
                    # remove current node from openset and add into closedset
                    openset.remove(current)
                    closedset.add(current)
                    
                    # loop through each neighbour
                    for neighbor in neighbor_nodes(current):
                        # ignore neighbor which is already evaluated.
                        if neighbor in closedset:
                            continue
                        # add new node to openset
                        if neighbor not in openset:
                            openset.add(neighbor)
                            
                        # The distance from start to a neighbor
                        tentative_g_score = g_score[current] + distance(current, neighbor)
                        
                        # if tentative g score is more than previous neighbour node g score, this is not a better path.
                        if tentative_g_score >= g_score.get(neighbor, float('inf')):
                            continue
                        
                        # resultant node will yield best path, save all info
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + cost_estimate(neighbor, goal)
                
                
                # st.text("cant find path")
                print("cant find path")
        
        # return empty path array if no paths found
        # st.text("no available paths found")
        print("no available paths found")
        return []

    #========================================================================== 


    startflag = True
    endflag = True
    resume = True

    # loops till user decides to quit
    while (resume):
    
        for label in directory_dict:
            # st.text(label, "-", directory_dict[label])
            st.text(label + ": -" + str(directory_dict[label]))

        # Create two columns
        col1, col2 = st.columns(2)
        # prompts user to input starting location
        start_loc_options = list(directory_dict.keys())
        start_loc = col1.radio("Please select your starting location (nearest shop to you)", start_loc_options)

        # prompts user to input starting location
        while (startflag):
            count = 0
            # timestamp = int(time.time())
            # start_loc = st.text_input("Please input your starting location (nearest shop to you) ", key=f"start_location_input_{timestamp}")
            for x in range(len(map_data)):
                count +=1
                
                # compares map label values and dictionary key values
                # if match, stores matching map label coordinates
                if map_data[x].description == directory_dict.get(start_loc):
                    st.text("Found starting location!")                
                    startflag = False
                    startx = map_data[x].bounding_poly.vertices[0].x
                    starty = map_data[x].bounding_poly.vertices[0].y
                    break
                
                elif x == len(map_data) - 1:
                    st.text("Starting location not found, please try again.")
            
        #prompts user to input destination goal        
        while (endflag):
            count = 0
            start_loc_options = list(directory_dict.keys())
            end_loc = col2.radio("Please input your intended destination:", start_loc_options)
            # end_loc = st.text_input("Please input your intended destination: ", key="end_location")
            for x in range(len(map_data)):
                count+=1
                
                # compares map label values and dictionary key values
                # if match, stores matching map label coordinates
                # then, inserts start and goal coordinates to obtain route
                if map_data[x].description == directory_dict.get(end_loc):
                    st.text("Found destination!")
                    endflag = False
                    endx = map_data[x].bounding_poly.vertices[0].x
                    endy = map_data[x].bounding_poly.vertices[0].y
                    
                    # returns and plots route between supplied coordinates in an output image
                    return_route(startx, starty, endx, endy)
                    
                    # read image to obtain dimensions
                    dimensions = cv2.imread("route.png",0)  
                    w, h = dimensions.shape
                    
                    # enlarge image
                    m_img = cv2.imread("route.png")  
                    m_img = cv2.resize(m_img, (int(h*size),int(w*size)), interpolation = cv2.INTER_CUBIC)
                    
                    # label starting and goal destination

                    cv2.putText(m_img,"Location", (startx,starty), cv2.FONT_HERSHEY_SIMPLEX, 2, 255,5, 2)
                    cv2.putText(m_img,"Destination", (endx,endy), cv2.FONT_HERSHEY_SIMPLEX, 2, 255,5, 2)
                    
                    # saves result image
                    # cv2.imwrite("route.png", m_img)
                    # saves result image
                    cv2.imwrite("route_with_labels.png", m_img)

                    # display route image in Streamlit app
                    st.image("route_with_labels.png", caption="Route from Starting Location to Destination", use_column_width=True)                                
                
                elif x == len(map_data) - 1:
                    st.text("Destination not found, please try again")
                    
        # # allows user to run navigation search again or quit program
        # again = st.text_input("Do you wish to continue? (Y/N): ", key="again")
        # resumestatus = True
        
        # # if yes, the whole while loop will execute once more
        # # if no, exits loop and program ends
        # while (resumestatus):
        #     if (again == "Y" or again == "y"):
        #         resumestatus = False
        #         startflag = True
        #         endflag = True
        #         #break
            
        #     elif (again == "N" or again == "n"):
        #         resumestatus = False
        #         resume = False
            
        #     # prompts user input again if invalid
        #     else:
        #         st.text("Invalid input, please try again")
        #         again = st.text_input("Do you wish to continue? (Y/N): " , key="invalid_again")
            
    st.text("      Good BYE")

page_names_to_funcs = {
    "Preprocessing": preprocessing,
    "Routing": routing
}

demo_name = st.sidebar.selectbox("Menu", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()



