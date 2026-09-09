import math
#import string
import numpy as np

#data input as [[[values],[lables]],[[values1],[lables1]],...]
def KNN(data, pointToFind, k):
  #find Closests
  #closest[0] = dist, [1] = value, [2] = labels, [3] = weight
  
  Closest = {"Dist": [], "Value": [], "Labels": [], "Weight": []}
  for Point in data:
    dist = 0
    for variableId in range(len(Point[0])):
      dist += ((pointToFind[variableId]-Point[0][variableId])**2)
    dist = math.sqrt(dist)
    
    #add to Closest
    if len(Closest["Dist"]) >= k:
      compareItem = max(Closest["Dist"])
      if compareItem > dist:
        #get indexs and choose the first one
        index = Closest["Dist"].index(compareItem)
        Closest["Dist"][index] = dist
        Closest["Value"][index] = Point[0]
        Closest["Labels"][index] = Point[1]
        Closest["Weight"][index] = [0]
    else:
      Closest["Dist"].append(dist)
      Closest["Value"].append(Point[0])
      Closest["Labels"].append(Point[1])
      Closest["Weight"].append([0])

  #Value fix/weighting
  MinimumVal = min(Closest["Dist"])
  MaximumVal = max(Closest["Dist"])

  #Comparaisons[0] = labels, Comparaisons[1] = total weight
  Comparaisons = {"Labels": [], "Weights": []}
  for itemIndex in range(len(Closest["Dist"])):
    value = (Closest["Dist"][itemIndex] - MinimumVal)
    if (MaximumVal-MinimumVal) != 0:
      value = value/(MaximumVal-MinimumVal)
    value = 1 - value
    Closest["Weight"][itemIndex] = value
  
    if Closest["Labels"][itemIndex] not in Comparaisons["Labels"]:
      Comparaisons["Labels"].append(Closest["Labels"][itemIndex]) #WARNING: I don't know why but 0 works here. If there is more than one label per thing it doesn't work
      Comparaisons["Weights"].append(Closest["Weight"][itemIndex])
       
    else:
      CompIndex = Comparaisons["Labels"].index(Closest["Labels"][itemIndex])
      Comparaisons["Weights"][CompIndex] += Closest["Weight"][itemIndex]
    
  
  
  retLabel = Comparaisons["Labels"][Comparaisons["Weights"].index(max(Comparaisons["Weights"]))] #from labels, the one with the highest weighted score

  return retLabel
file_path = "iris_example_project/Iris.csv"
data = np.genfromtxt(file_path, delimiter=',', usecols=(0,1,2,3,4), skip_header=1)
species = np.genfromtxt(file_path, delimiter=',', usecols=(5), dtype = str, skip_header=1)
print(species)
