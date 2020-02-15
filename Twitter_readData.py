import os
import numpy as np
import pickle
import json
import string
import networkx as nx
import time
from numpy.core.multiarray import dtype
import operator
import nltk.data
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def getint(name):
    basename = name.partition('.')
    
    return int(basename[0])

def generateUserList():
    folderName = "C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\Tweeter\\Tweets\\tweets\\"
    users = np.asarray(map(int, os.listdir(folderName)))
    pickle.dump(users, open("users.p", "wb"))
    
def getUserList():
    return np.sort(pickle.load(open("users.p", "rb")).astype(int))

def getNetwork():
    otputFile = open('networkx.txt', 'w')
    users = getUserList()
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\Tweeter\\network.txt'
    f = open(fileName)
    for line in iter(f):
        if line.strip():
            line_splitted = line.split('\t')
            userId1 = int(line_splitted[0])
            userId2 = int(line_splitted[1])
            if (np.any(users == userId1)):
                if(np.any(users == userId2)):
                    otputFile.write(line)
        else:
            print "Missing Value Empty"
def getUsers():
    otputFile = open('users.txt', 'w')
    users = getUserList()
    fileName = 'C:\Users\Kazi Abir Adnan\Desktop\Datasets\Tweeter\users.txt'
    f = open(fileName)
    for line in iter(f):
        if line.strip():
            line_splitted = line.split('\t')
            if(line_splitted.__len__() >= 8):
                userId = line_splitted[0]
                if userId:
                    userId = int(userId)
                    if np.any(users == userId):
                        otputFile.write(line)
                else:
                    print "Missing Value for invalid id " + userId
            else:
                print "Missing Value for less inputs of size" + str(line_splitted.__len__())
        else:
            print "Missing Value Empty"
    otputFile.close()

def createAdjacencyMatrix():
    fileName = 'network.txt'
    adjacencyMatrix = dict()
    f = open(fileName)
    for line in iter(f):
        line_splitted = line.split('\t')
        user1 = int(line_splitted[0])
        user2 = int(line_splitted[1])
        print user1
        if user1 in adjacencyMatrix:
            list = adjacencyMatrix[user1]
            if user2 not in list: 
                adjacencyMatrix[user1].append(user2)
        else:
            adjacencyMatrix[user1] = [user2]
        if user2 in adjacencyMatrix:
            list = adjacencyMatrix[user2]
            if user1 not in list: 
                adjacencyMatrix[user2].append(user1)
        else:
            adjacencyMatrix[user2] = [user1]
    
    pickle.dump(adjacencyMatrix, open("adjacencyMatrix.p", "wb"))   

def createTweetMatrix():
    tweetMatrix = dict()
    folderName = "C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\Tweeter\\Tweets\\tweets"
    file_names = os.listdir(folderName)
    file_names.sort(key=getint)
    for filename in file_names:
        userId = int(filename) 
        tweetMatrix[userId] = []
        f = open(folderName+"\\"+filename)
        f.readline()
        data = f.read()
        tweets = data.split("***\n***\n")
        print userId
        hashtagDic = dict()
        for tweet in tweets:
            tweet = tweet.strip()
            lines = tweet.split('\n')
            for line in lines :   
                if (line.startswith( 'ID:' )):
                    id = line.split(':')[1]
                    if not id:
                        continue
                    try:
                        tweetId = int(id.strip())
                    except ValueError:
                        continue
                    tweetId = int(id.strip())
                if (line.startswith( 'Hashtags:' )):
                    hashtagLine = line.split(':')
            hashtags = hashtagLine[1]
            if hashtags:
                hashtags = hashtags.strip()
                hashtags = hashtags.split(" ")
                if hashtags.__len__()>1:
                    hashtagDic[tweetId] = []
                    for hashtag in hashtags:
                        hashtagDic[tweetId].append(hashtag)
                else:
                   hashtagDic[tweetId] =  hashtags[0]
            else:
                hashtagDic[tweetId] = ''
        tweetMatrix[userId].append(hashtagDic)
    pickle.dump(tweetMatrix, open("tweetMatrix.p", "wb"))

def readTweetMatrix():
    otputFile = open('hastagCount.txt', 'w')
    folderName = "C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\Tweeter\\Tweets\\tweets"
    file_names = os.listdir(folderName)
    file_names.sort(key=getint)
    for filename in file_names:
        userId = int(filename) 
        otputFile.write(str(userId)+"\t")
        f = open(folderName+"\\"+filename)
        f.readline()
        data = f.read()
        tweets = data.split("***\n***\n")
        print userId
        count = 1
        for tweet in tweets:
            tweet = tweet.strip()
            lines = tweet.split('\n')
            for line in lines :
                if (line.startswith( 'Hashtags:' )):
                    hashtagLine = line.split(':')
            hashtags = hashtagLine[1]
            if hashtags:
                hashtags = hashtags.strip()
                hashtags = hashtags.split(" ")
                otputFile.write(str(hashtags.__len__()))
            else:
                otputFile.write(str(0))
            if (count != len(tweets)):
                otputFile.write(' ')
            count = count + 1  
        otputFile.write('\n')
    otputFile.close()
    
    return 0

def stats():
    fileName = 'hastagCount.txt'
    hashTagTotalCount = dict()
    f = open(fileName)
    for line in iter(f):
        line_splitted = line.split('\t')
        counts = line_splitted[1].split(' ')
        for count in counts:
            hashTagCount = int(count)
            if hashTagCount in hashTagTotalCount:
                temp = hashTagTotalCount[hashTagCount]
                temp = temp + 1
                hashTagTotalCount[hashTagCount] = temp
            else:
                hashTagTotalCount[hashTagCount] = 1
        print line_splitted[0]    
                
    for key,val in hashTagTotalCount.items():
        print key, "=>", val
        
def createGraph():
    usersOrig = np.sort(np.loadtxt('usersinnetwork.txt',dtype=np.int32))
    print "User loaded"
    users = dict()
    usersRev = dict()
    nodes = len(usersOrig)
    for i in range(nodes):
        users[i] = usersOrig[i]
        usersRev[usersOrig[i]] = i
    print "Hashmap created"
    G1 = nx.Graph()
    for i in range(nodes):
        G1.add_node(i)
    print "Node added to graph"
    f = open('networkx.txt')
    adjacencyMatrix = dict()
    for line in iter(f):
        if line.strip():
            line_splitted = line.split('\t')
            user1 = usersRev[int(line_splitted[0])]
            user2 = usersRev[int(line_splitted[1])]
            print users[user1]
            if user1 in adjacencyMatrix:
                list1 = adjacencyMatrix[user1]
                if user2 not in list1: 
                    adjacencyMatrix[user1].append(user2)
                    G1.add_edge(user1, user2, weight=1)
            else:
                adjacencyMatrix[user1] = [user2]
                G1.add_edge(user1, user2, weight=1)
               
            if user2 in adjacencyMatrix:
                list1 = adjacencyMatrix[user2]
                if user1 not in list1: 
                    adjacencyMatrix[user2].append(user1)
            else:
                adjacencyMatrix[user2] = [user1]
    return G1
def hashtagDictionary():
    cachedStopWords = stopwords.words("english")
    ps = PorterStemmer()
    folderName = "C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\Tweeter\\Tweets\\tweets"
    file_names = os.listdir(folderName)
    file_names.sort(key=getint)
    hashtagDic = dict()
    for filename in file_names:
        userId = int(filename) 
        f = open(folderName+"\\"+filename)
        f.readline()
        data = f.read()
        tweets = data.split("***\n***\n")
        print userId
        for tweet in tweets:
            tweet = tweet.strip()
            lines = tweet.split('\n')
            for line in lines :
                if (line.startswith( 'Hashtags:' )):
                    hashtagLine = line.split(':')
            hashtags = hashtagLine[1]
            if hashtags:
                hashtags = hashtags.strip()
                hashtags = hashtags.split(" ")
                if hashtags.__len__()>1:
                    for hashtag in hashtags:
                        hashtag = hashtag.strip()
                        hashtag = re.sub(r'\W+', '', hashtag).lower()
                        if hashtag in hashtagDic:
                            hashtagDic[hashtag] = hashtagDic[hashtag] + 1
                        else:
                            hashtagDic[hashtag] = 1
                else:
                    hashtag = hashtags[0].strip()
                    hashtag = re.sub(r'\W+', '', hashtag).lower()
                    if hashtag in hashtagDic:
                        hashtagDic[hashtag] = hashtagDic[hashtag] + 1
                    else:
                        hashtagDic[hashtag] = 1
    otputFile = open('hashtagDictionary.txt', 'w')
    otputFile.write(str(len(hashtagDic))+"\n")
    sorted_x = sorted(hashtagDic.items(), key=operator.itemgetter(1), reverse=True)
    for item in sorted_x:
        otputFile.write(str(item[0])+"\t"+str(item[1])+"\n")
    otputFile.close()
def tweetTime():
    timestamp = dict()
    folderName = "C:\\Users\\kadnan\\Desktop\\Twitter\\tweets"
    file_names = os.listdir(folderName)
    file_names.sort(key=getint)
    for filename in file_names:
        userId = int(filename) 
        f = open(folderName+"\\"+filename)
        f.readline()
        data = f.read()
        tweets = data.split("***\n***\n")
        print userId
        for tweet in tweets:
            tweet = tweet.strip()
            lines = tweet.split('\n')
            for line in lines :
                if (line.startswith( 'Time:' )):
                    line = line.strip().split(' ')
                    if(len(line)==7):
                        year = ''
                        month = ''
                        try:
#                             print line
                            year = line[len(line)-1].strip()
                            month = line[2].strip()
                        except ValueError:
                            continue
                        time = year + '-' + month
                        if time in timestamp:
                            timestamp[time] = timestamp[time] + 1
                        else:
                            timestamp[time] = 1
    pickle.dump(timestamp, open("tweetYears.p", "wb"))

def tweetyears():
    years = pickle.load(open("tweetYears.p", "rb"))
    otputFile = open('tweetYears.txt', 'w')
    for time in sorted(years.keys(), reverse=True):
        wr = str(time)+':'+str(years[time])
        otputFile.write(wr+'\n')
    otputFile.close()
    
def yearHashtag():
    hashtagDict = dict()
    f = open('tweetYears.txt')
    for line in f.readlines():
        elements = line.split(':')
        time = elements[0].strip()
        hashtagDict[time] = dict()
    f = open('hashtagDictionary.txt')
    f.readline()
    totalhashtags = dict()
    for line in f.readlines():
        elements = line.split('\t')
        hashtag = elements[0].strip()
        freq = int(elements[1].strip())
        totalhashtags[hashtag] = freq
    
    folderName = "C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\Tweeter\\Tweets\\tweets"
    file_names = os.listdir(folderName)
    file_names.sort(key=getint)
    for filename in file_names:
        userId = int(filename) 
        f = open(folderName+"\\"+filename)
        f.readline()
        data = f.read()
        tweets = data.split("***\n***\n")
        print userId
        for tweet in tweets:
            tweet = tweet.strip()
            lines = tweet.split('\n')
            for line in lines :
                if (line.startswith( 'Time:' )):
                    time = '' 
                    line = line.strip().split(' ')
                    if(len(line)==7):
                        year = ''
                        month = ''
                        try:
                            year = line[len(line)-1].strip()
                            month = line[2].strip()
                        except ValueError:
                            continue
                        time = year + '-' + month
                elif (line.startswith( 'Hashtags:' )):
                    hashtagLine = line.split(':')
            hashtags = hashtagLine[1]
            if ((time and hashtags) and time in hashtagDict):
                hashtags = hashtags.strip()
                hashtags = hashtags.split(" ")
                if hashtags.__len__()>1:
                    for hashtag in hashtags:
                        hashtag = hashtag.strip()
                        hashtag = re.sub(r'\W+', '', hashtag).lower()
                        data = hashtagDict[time]
                        if hashtag in data:
                            data[hashtag] = data[hashtag] + 1
                        else:
                            data[hashtag] = 1
                else:
                    hashtag = hashtags[0].strip()
                    hashtag = re.sub(r'\W+', '', hashtag).lower()
                    data = hashtagDict[time]
                    if hashtag in data:
                        data[hashtag] = data[hashtag] + 1
                    else:
                        data[hashtag] = 1
    
    otputFile = open('hashtag_distribution.txt', 'w')
    for year, val in hashtagDict.iteritems():
        otputFile.write(year+'\t')
        sorted_x = sorted(val.items(), key=operator.itemgetter(1), reverse=True)
        for item in sorted_x:
            keyword = str(item[0])
            freq = int(item[1])
            otputFile.write(keyword+':'+str(freq)+',')
        otputFile.write('\n')
    otputFile.close()

def dist():
    wordDict = dict()
    yearsMap = dict()
    yearsMapRev = dict()
    hashtagDict1 = dict()
    f = open('hashtag_distribution_short.txt')
    for line in f.readlines():
        elements = line.split('\t')
        time = elements[0].strip()
        hashtagDict1[time] = elements[1].strip()
    count = 0
    hashtagDict = dict()
    dates = hashtagDict1.keys()
    dates_list = list()
    for date in dates:
        dates_list.append(datetime.datetime.strptime(date, '%Y-%b-%d').date())
    for date in sorted(dates_list):
        hashtagDict[date] = dict()
        yearsMap[count] = date
        yearsMapRev[date] = count
        count  = count + 1
        
    f = open('hashtag_distribution_short.txt')
    for line in f.readlines():
        elements = line.split('\t')
        year = elements[0].strip()
        year = datetime.datetime.strptime(year, '%Y-%b-%d').date()
        print year
        words = elements[1].strip()
        if words:
            words = words.strip().split(',')
            for word in words:
                if word:
                    elements = word.split(':')
                    keyword = elements[0].strip()
                    freq = int(elements[1].strip())
                    if (keyword and freq>100):
                        if (keyword in wordDict):
                            years = wordDict[keyword]
                            years[year] = freq
                        else:
                           wordDict[keyword] = dict()
                           years = wordDict[keyword]   
                           years[year] = freq
    data = np.zeros((len(wordDict), len(hashtagDict)))
    keywordsMap = dict()
    keywordsMapRev = dict()
    count = 0
    for word in sorted(wordDict.iterkeys()):
        keywordsMap[count] = word
        keywordsMapRev[word] = count
        count  = count + 1
    for word in sorted(wordDict.iterkeys()):
        for year in range(len(hashtagDict)):
            dictionary = wordDict[word]
            y = yearsMap[year]
            if y in dictionary:
                freq = dictionary[y]
                print freq
                data[keywordsMapRev[word], year] = freq
            else:
                data[keywordsMapRev[word],year] = 0
    
    normed_matrix = normalize(data, axis=1, norm='l1')
    plt.imshow(normed_matrix, aspect='auto', interpolation='none', origin='lower', cmap=plt.get_cmap('hot'))
    plt.tick_params(labeltop=True, labelright=True)
    plt.colorbar()
    sorted_x = sorted(keywordsMapRev.items(), key=operator.itemgetter(1))
    sorted_x = [i[0] for i in sorted_x]
    plt.yticks(range(len(keywordsMapRev)),sorted_x, size='small')
    sorted_y = sorted(yearsMapRev.items(), key=operator.itemgetter(1))
    sorted_y = [i[0] for i in sorted_y]
    plt.xticks(range(len(yearsMapRev)),sorted_y, size='small',rotation='vertical')
    plt.show()
    
def reduce():
    otputFile = open('hashtag_distribution_short.txt', 'w')
    f = open('hashtag_distribution.txt')
    for line in f.readlines():
        elements = line.split('\t')
        year = elements[0].strip()
        otputFile.write(year+'\t')
        words = elements[1].strip()
        if words:
            count = 0
            words = words.split(',')
            for word in words:
                if (count<11):
                    elements = word.split(':')
                    keyword = elements[0].strip()
                    freq = int(elements[1].strip())
                    if count<10:
                        otputFile.write(keyword+':'+str(freq)+',')
                    else:
                        otputFile.write(keyword+':'+str(freq))
                    count = count + 1
                else:
                    break
        otputFile.write('\n')
    otputFile.close()
def main():
#     reduce()
    dist()
#     tweetyears()
#     tweetTime()
#     hashtagDictionary()
#     G1 = createGraph()
#     nodes = G1.number_of_nodes()
#     edges = G1.number_of_edges()
#     print "\tNumber of Nodes = ", nodes, " , Number of Edges = ", edges
#     print "\tDiameter is ", nx.diameter(G1);
#     start_time = time.time()
#     pickle.dump(G1, open("graph.p", "wb"))
#     print "Time to dump", (time.time()-start_time),"seconds"
if __name__ == '__main__':
    main()
