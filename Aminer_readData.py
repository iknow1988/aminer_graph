import os
import operator
import numpy as np
from numpy.core.multiarray import dtype
import networkx as nx
import time
import pickle
import sys
from sklearn.feature_extraction.text import CountVectorizer
import nltk
# nltk.download()
import nltk.data
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from mpltools import color
from sklearn.preprocessing import normalize

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())  

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("Aminer.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
def generateAuthorList():
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    authorMap = dict()
    count =1
    for paper in papers:
        print count
        lines = paper.split('\n')
        for line in lines:
            if (line.startswith( '#@' )):
                authors = line.split('@')[1]
                authors = authors.split(',')
                for author in authors:
                    if author not in authorMap:
                        authorMap[author]=1
                    else:
                        temp = authorMap[author]
                        temp = temp + 1
                        authorMap[author] = temp
        count = count + 1
    otputFile = open('authors.txt', 'w')
    otputFile.write(str(len(authorMap))+"\n")
    for key in sorted(authorMap.iterkeys()):
        otputFile.write(str(key)+"\t"+str(authorMap[key])+"\n")
    otputFile.close()

def generateYearsList():
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    yearsMap = dict()
    count =1
    for paper in papers:
#         print count
        lines = paper.split('\n')
        year = -1
        index = -1
        for line in lines:
            if (line.startswith( '#t' )):
                year = line.split('#t')[1].strip()
                year = int(year)
            elif (line.startswith( '#index' )):
                index = line.split('#index')[1].strip()
                index = int(index)
            else:
                continue
        if index not in yearsMap:
            yearsMap[index]=year
        else:
            print "here"
        count = count + 1
        
    otputFile = open('years.txt', 'w')
    otputFile.write(str(len(yearsMap))+"\n")
    for key in sorted(yearsMap.iterkeys()):
        otputFile.write(str(key)+"\t"+str(yearsMap[key])+"\n")
    otputFile.close()

def generateVenuesList():
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    venuesMap = dict()
    count =1
    for paper in papers:
#         print count
        lines = paper.split('\n')
        venue = ""
        index = -1
        for line in lines:
            if (line.startswith( '#c' )):
                temp = line.split('#c')
                if(len(temp)>1):
                    if temp[1]:
                        venue = temp[1].strip()
            elif (line.startswith( '#index' )):
                index = line.split('#index')[1].strip()
                index = int(index)
            else:
                continue
        if index not in venuesMap:
            venuesMap[index]=venue
        else:
            print "here"
        count = count + 1
        
    otputFile = open('venues.txt', 'w')
    otputFile.write(str(len(venuesMap))+"\n")
    for key in sorted(venuesMap.iterkeys()):
        otputFile.write(str(key)+"\t"+str(venuesMap[key])+"\n")
    otputFile.close()
    
def generateVenueCount():
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    venuesMap = dict()
    count =1
    for paper in papers:
        print count
        lines = paper.split('\n')
        for line in lines:
            if (line.startswith( '#c' )):
                temp = line.split('#c')
                if(len(temp)>1):
                    if temp[1]:
                        venue = temp[1].strip()
                        if venue not in venuesMap:
                            venuesMap[venue]=1
                        else:
                            temp = venuesMap[venue]
                            temp = temp + 1
                            venuesMap[venue] = temp
                    else:
                        venue = ""
                        if venue not in venuesMap:
                            venuesMap[venue]=1
                        else:
                            temp = venuesMap[venue]
                            temp = temp + 1
                            venuesMap[venue] = temp
            else:
                continue
        count = count + 1
    otputFile = open('venuesCount.txt', 'w')
    otputFile.write(str(len(venuesMap))+"\n")
    for key in sorted(venuesMap.iterkeys()):
        otputFile.write(str(key)+"\t"+str(venuesMap[key])+"\n")
    otputFile.close()

def generateVenueCountbyValues():
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    venuesMap = dict()
    count =1
    for paper in papers:
        print count
        lines = paper.split('\n')
        for line in lines:
            if (line.startswith( '#c' )):
                temp = line.split('#c')
                if(len(temp)>1):
                    if temp[1]:
                        venue = temp[1].strip()
                        if venue not in venuesMap:
                            venuesMap[venue]=1
                        else:
                            temp = venuesMap[venue]
                            temp = temp + 1
                            venuesMap[venue] = temp
                    else:
                        venue = ""
                        if venue not in venuesMap:
                            venuesMap[venue]=1
                        else:
                            temp = venuesMap[venue]
                            temp = temp + 1
                            venuesMap[venue] = temp
            else:
                continue
        count = count + 1
    otputFile = open('venuesCountbyValues.txt', 'w')
    otputFile.write(str(len(venuesMap))+"\n")
    sorted_x = sorted(venuesMap.items(), key=operator.itemgetter(1))
    for item in sorted_x:
        otputFile.write(str(item[0])+"\t"+str(item[1])+"\n")
    otputFile.close()
    
def generateReferenceList():
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    referenceMap = dict()
    count =1
    for paper in papers:
        print count
        index = -1
        references = list()
        lines = paper.split('\n')
        for line in lines:
            if (line.startswith( '#%' )):
                temp = line.split('#%')
                if(len(temp)>1):
                    if temp[1]:
                        pap = int(temp[1].strip())
                        references.append(pap)
            elif (line.startswith( '#index' )):
                index = line.split('#index')[1].strip()
                index = int(index)
            else:
                continue
        if index not in referenceMap:
            referenceMap[index]=references
        count = count + 1
    otputFile = open('references.txt', 'w')
    otputFile.write(str(len(referenceMap))+"\n")
    for key in sorted(referenceMap.iterkeys()):
        otputFile.write(str(key)+"\t")
        rr = referenceMap[key]
        length = len(rr)
        if(length ==0):
            otputFile.write(str(-1)+"\n")
        elif(length ==1):
            otputFile.write(str(rr[0])+"\n")
        else: 
            for i in range(length):
                if(i<length-1):
                    otputFile.write(str(rr[i])+",")
                else:
                    otputFile.write(str(rr[i]))
            otputFile.write("\n")
    otputFile.close()

def generateReferenceCountList():
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    referenceMap = dict()
    count =1
    for paper in papers:
        print count
        index = -1
        references = list()
        lines = paper.split('\n')
        for line in lines:
            if (line.startswith( '#%' )):
                temp = line.split('#%')
                if(len(temp)>1):
                    if temp[1]:
                        pap = int(temp[1].strip())
                        references.append(pap)
            elif (line.startswith( '#index' )):
                index = line.split('#index')[1].strip()
                index = int(index)
            else:
                continue
        if index not in referenceMap:
            referenceMap[index]=len(references)
        count = count + 1
    otputFile = open('referencesCount.txt', 'w')
    otputFile.write(str(len(referenceMap))+"\n")
    sorted_x = sorted(referenceMap.items(), key=operator.itemgetter(1))
    for item in sorted_x:
        otputFile.write(str(item[0])+"\t"+str(item[1])+"\n")
    otputFile.close()

def generateYearsCount():
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    yearsMap = dict()
    count =1
    for paper in papers:
#         print count
        lines = paper.split('\n')
        year = -1
        for line in lines:
            if (line.startswith( '#t' )):
                year = line.split('#t')[1].strip()
                year = int(year)
            else:
                continue
        if year not in yearsMap:
            yearsMap[year]=1
        else:
            yearsMap[year] = yearsMap[year] + 1
        count = count + 1
        
    otputFile = open('years.txt', 'w')
    otputFile.write(str(len(yearsMap))+"\n")
    tmp =0 
    for key in sorted(yearsMap.iterkeys()):
        otputFile.write(str(key)+"\t"+str(yearsMap[key])+"\n")
        tmp = tmp + int(yearsMap[key])
    print tmp
    otputFile.close()

def generateCitationList():
    fileName = 'C:\\Users\\kadnan\\Desktop\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    authorMap = dict()
    count =1
    for paper in papers:
        print count
        lines = paper.split('\n')
        for line in lines:
            if (line.startswith( '#@' )):
                authors_sorted = line.split('@')[1]
                authors_sorted = authors_sorted.split(',')
                for author in authors_sorted:
                    if author not in authorMap:
                        authorMap[author]=1
                    else:
                        temp = authorMap[author]
                        temp = temp + 1
                        authorMap[author] = temp
        count = count + 1
    authors_sorted = sorted(authorMap.iterkeys())
    totalAuthor = len(authors_sorted)
    authors = dict()
    authorsRev = dict()
    for i in range(totalAuthor):
        authors[i] = authors_sorted[i]
        authorsRev[authors_sorted[i]] = i
    print "Hashmap created"
#     otputFile = open('map.txt', 'w')
#     for key,val in authors.items():
#         otputFile.write(str(key)+"\t"+str(val)+"\n")
#     otputFile.close()
    return authors, authorsRev

def generateAuthorGraph(authors, authorsRev):
    G1 = nx.Graph()
    nodes = len(authors)
    for i in range(nodes):
        G1.add_node(i)
    print "Node added to graph"
    adjacencyMatrix = dict()
    fileName = 'C:\\Users\\kadnan\\Desktop\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    count =1
    for paper in papers:
        lines = paper.split('\n')
        for line in lines:
            if (line.startswith( '#@' )):
                authors = line.split('@')[1]
                authors = authors.split(',')
                length = len(authors)
                if(length==1):
                   author1 = authors[0]
                   author1 = authorsRev[author1]
                   if author1 not in adjacencyMatrix:
                       adjacencyMatrix[author1]=dict()
                else:
                    for i in range(length):
                        for j in range(i+1,length):
                            author1 = authorsRev[authors[i]]
                            author2 = authorsRev[authors[j]]
                            if author1 in adjacencyMatrix:
                                lst = adjacencyMatrix[author1]
                                if author2 not in lst:
                                    lst[author2] = 1
                                else:
                                    lst[author2] = lst[author2] + 1
                            else:
                                adjacencyMatrix[author1]=dict()
                                lst = adjacencyMatrix[author1]
                                lst[author2] = 1
                            
                            if author2 in adjacencyMatrix:
                                lst = adjacencyMatrix[author2]
                                if author1 not in lst:
                                    lst[author1] = 1
                                else:
                                    lst[author1] = lst[author1] + 1
                            else:
                                adjacencyMatrix[author2]=dict()
                                lst = adjacencyMatrix[author2]
                                lst[author1] = 1
        count = count + 1
#     otputFile = open('adjacencyMatrix.txt', 'w')
#     for author1 in sorted(adjacencyMatrix.iterkeys()):
#         otputFile.write(str(author1)+'\t')
#         authors = adjacencyMatrix[author1]
#         length = len(authors)
#         count = 0
#         for key in sorted(authors.iterkeys()):
#             author2 = key
#             if(count<length-1):
#                 otputFile.write(str(author2)+',')
#             else:
#                 otputFile.write(str(author2))
#             count = count + 1
#         otputFile.write('\n')
#     otputFile.close()      
    for author1 in sorted(adjacencyMatrix.iterkeys()):
        authors = adjacencyMatrix[author1]
        length = len(authors)
        for key in sorted(authors.iterkeys()):
            author2 = key
            if not G1.has_edge(author1, author2):
                G1.add_edge(author1, author2, weight=1)
    start_time = time.time()           
    pickle.dump(G1, open("Aminer_Graph.p", "wb"))   
    print "Time to save", (time.time()-start_time),"seconds"     
    
def readGraph():
    start_time = time.time()
    G1 = pickle.load(open("Aminer_Graph.p", "rb"))
    print "Time to load", (time.time()-start_time),"seconds"
    
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    
    print "nodes =",nodes, ", Edges =", edges
    print "Is connected = ", nx.is_connected(G1)
    print "Number of connected components = ",  nx.number_connected_components(G1)
    Gc = max(nx.connected_component_subgraphs(G1), key=len)
    print "nodes =",Gc.number_of_nodes(), ", Edges =", Gc.number_of_edges()
    
def checkGraph():  
    start_time = time.time()
    G1 = pickle.load(open("Aminer_Graph.p", "rb"))
    print "Time to load", (time.time()-start_time),"seconds"
     
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
#     otputFile = open('degree.txt', 'w')
#     for NI in range(nodes):
#         deg = G1.degree(NI)
#         otputFile.write(str(NI)+'\t'+str(deg)+'\n')
#     otputFile.close()
    degrees = np.zeros((nodes))
    for NI in range(nodes):
        deg = G1.degree(NI)
        degrees[NI] = deg
    degreessorted = np.sort(degrees)
    np.savetxt('myfile.txt', degrees.transpose(),fmt='%i')
    np.savetxt('myfile1.txt', degreessorted.transpose(),fmt='%i')
    
def graphDiameter():
    start_time = time.time()
    G1 = pickle.load(open("Aminer_Graph.p", "rb"))
    print "Time to load", (time.time()-start_time),"seconds"
    if (nx.is_connected(G1)):
        diameter = nx.diameter(G1)
        print "\tDiameter is ", diameter
    else:
        print nx.number_connected_components(G1)
        graphs = list(nx.connected_component_subgraphs(G1))
        for i in range(len(graphs)):
            start_time = time.time() 
            print i,"\tDiameter is ", nx.diameter(graphs[i])
            print "Time to count diameter", (time.time()-start_time),"seconds"
def getDocuments():
    fileName = 'C:\\Users\\Kazi Abir Adnan\\Desktop\\Datasets\\Aminer\\outputacm.txt'
    f = open(fileName)
    line = f.readline().strip()
    size = int(line)
    data = f.read()
    papers = data.split("\n\n")
    count =1
    documents = list()
    for paper in papers:
        print count
        lines = paper.split('\n')
        for line in lines:
            if (line.startswith( '#*' )):
                content = line.split('#*')[1].strip()
                documents.append(content)
        count = count + 1
        
    return documents

def getDictionary(documents):
    cachedStopWords = stopwords.words("english")
    ps = PorterStemmer()
    vocab = dict()
    count = 0
    for text in documents:
        print count
        for word in text.split():
            word = word.strip()
            word = re.sub(r'\W+', '', word).lower()
            if word not in cachedStopWords:
                if word:
                    word = ps.stem(word)
                    if word in vocab:
                        vocab[word] = vocab[word] + 1
                    else:
                        vocab[word] = 1
        count = count + 1
    otputFile = open('titles.txt', 'w')
    otputFile.write(str(len(vocab))+"\n")
    sorted_x = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    for item in sorted_x:
        otputFile.write(str(item[0])+"\t"+str(item[1])+"\n")
    otputFile.close()
    
def keywordDistibution():
    cachedStopWords = stopwords.words("english")
    f = open('titles.txt')
    ps = PorterStemmer()
    f.readline()
    keywords = dict()
    for line in f.readlines():
        line = line.split('\t')
        keyword = line[0]
        frequency = int(line[1])
        keywords[keyword] = frequency
    
    years = dict()
    f = open('years_paper_count.txt')
    f.readline()
    f.readline()
    for line in f.readlines():
        year = int(line.split('\t')[0])
        years[year] = dict()
    
    fileName = 'C:\\Users\\kadnan\\Desktop\\Aminer\\outputacm.txt'
    f = open(fileName)
    f.readline()
    data = f.read()
    papers = data.split("\n\n")
    count =1
    for paper in papers:
        print count
        lines = paper.split('\n')
        year = -1
        text = ''
        for line in lines:
            if (line.startswith( '#t' )):
                year = line.split('#t')[1].strip()
                year = int(year)
            elif (line.startswith( '#*' )):
                text = line.split('#*')[1].strip()
            else:
                continue
        if ((year in years) and text):
            for word in text.split():
                word = word.strip()
                word = re.sub(r'\W+', '', word).lower()
                word = ps.stem(word)
                if word not in cachedStopWords:
                    if word in keywords:
                        data = years[year]
                        if word in data:
                            data[word] = data[word] + 1
                        else:
                            data[word] = 1
        count = count + 1
    pickle.dump(years, open("keyword_distribution.p", "wb"))
    
def readKeywordDistibution():
    f = open('titles.txt')
    f.readline()
    keywords = list()
    keywordsMap = dict()
    keywordsMapRev = dict()
    count = 0
    for line in f.readlines():
        line = line.split('\t')
        keyword = line[0]
        frequency = int(line[1])
        if (frequency>999):
            keywords.append(keyword)
            keywordsMap[count] = keyword
            keywordsMapRev[keyword] = count
            count = count + 1
    
    years = list()
    yearsMap = dict()
    yearsMapRev = dict()
    count = 0
    f = open('years_paper_count.txt')
    f.readline()
    f.readline()
    for line in f.readlines():
        year = int(line.split('\t')[0])
        if(year>1989):
            years.append(year)
            yearsMap[count] = year
            yearsMapRev[year] = count
            count = count + 1
    
    dist = np.zeros((len(years),len(keywords)))
    
    distribution = pickle.load(open("keyword_distribution.p", "rb"))
    for year, val in distribution.iteritems():
        for keyword, freq in val.iteritems():
            dist[yearsMapRev[year],keywordsMapRev[keyword]] = freq
    pickle.dump(dist, open("keyword_distribution_image.p", "wb"))
    
def plotKeywordDistibution():
    f = open('titles.txt')
    f.readline()
    keywords = dict()
    keywordsMap = dict()
    keywordsMapRev = dict()
    count = 0
    for line in f.readlines():
        line = line.split('\t')
        keyword = line[0]
        frequency = int(line[1])
        keywords[keyword] = frequency
        keywordsMap[count] = keyword
        keywordsMapRev[keyword] = count
        count = count + 1
    
    years = list()
    yearsMap = dict()
    yearsMapRev = dict()
    count = 0
    f = open('years_paper_count.txt')
    f.readline()
    f.readline()
    for line in f.readlines():
        year = int(line.split('\t')[0])
        if((year>1979) and (year<2010)):
            years.append(year)
            yearsMap[count] = year
            yearsMapRev[year] = count
            count = count + 1
    distribution = pickle.load(open("keyword_distribution.p", "rb"))
    dist = np.zeros((len(years),len(keywords)))
    for year, val in distribution.iteritems():
        count = 1
        if ((year>1979) and (year<2010)):
            print year,'\t',
            sorted_x = sorted(val.items(), key=operator.itemgetter(1), reverse=True)
            for item in sorted_x:
                if (count< 31):
                    keyword = item[0]
                    freq = int(item[1])
                    y = yearsMapRev[year]
                    k = keywordsMapRev[keyword]
                    dist[y,k] = freq
                    if(count<30):
                        print keyword,':',freq,',',
                    else:
                        print keyword,':',freq,
                    count = count + 1
                else:
                    break
            print ''
#     distribution = pickle.load(open("keyword_distribution_image.p", "rb")).astype(int)
#     plt.imshow(dist.transpose(), aspect='auto', interpolation='none', origin='lower')
#     plt.colorbar()
#     plt.xticks(range(len(yearsMapRev)),yearsMapRev.keys(), size='small')
#     sorted_x = sorted(keywordsMapRev.items(), key=operator.itemgetter(1))
#     sorted_x = [i[0] for i in sorted_x]
#     plt.yticks(range(len(keywordsMapRev)),sorted_x, size='small')
    plt.show()
def dist():
    prohitbit = list()
    prohitbit.append('1st')
    prohitbit.append('2000')
    prohitbit.append('2003')
    prohitbit.append('2004')
    prohitbit.append('2005')
    prohitbit.append('2006')
    prohitbit.append('2nd')
    f = open('dist.txt')
    wordDict = dict()
    for line in f.readlines():
        elements = line.split('\t')
        year = int(elements[0].strip())
        words = elements[1]
        words = words.split(',')
        for word in words:
            elements = word.split(':')
            keyword = elements[0].strip()
            freq = int(elements[1].strip())
            if keyword not in prohitbit:
                if keyword in wordDict:
                    years = wordDict[keyword]
                    years[year] = freq
                else:
                   wordDict[keyword] = dict()
                   years = wordDict[keyword]   
                   years[year] = freq
    data = np.zeros((len(wordDict), 30))
    keywordsMap = dict()
    keywordsMapRev = dict()
    count = 0
    for word in sorted(wordDict.iterkeys()):
        if word not in prohitbit:
            keywordsMap[count] = word
            keywordsMapRev[word] = count
            count  = count + 1
    for word in sorted(wordDict.iterkeys()):
        if word not in prohitbit:
            for year in range(30):
                dictionary = wordDict[word]
                y = year + 1980
                if y in dictionary:
                    freq = dictionary[y]
                    data[keywordsMapRev[word], year] = freq
                else:
                    data[keywordsMapRev[word],year] = 0
    normed_matrix = normalize(data, axis=1, norm='l1')
    plt.imshow(normed_matrix, aspect='auto', interpolation='none', origin='lower', cmap=plt.get_cmap('hot'))
    plt.colorbar()
    plt.tick_params(labeltop=True, labelright=True)
    sorted_x = sorted(keywordsMapRev.items(), key=operator.itemgetter(1))
    sorted_x = [i[0] for i in sorted_x]
    plt.yticks(range(len(keywordsMapRev)),sorted_x, size='small')
    plt.xticks(np.arange(0,30,1),np.arange(1980,2010,1), size='small')
    plt.show()
#         plt.plot(data,label=word, linewidth=2.0)
#         plt.xticks(np.arange(0,30,1),np.arange(1980,2010,1), size='small')
#         plt.legend(bbox_to_anchor=(1.02, 1.12), loc=2, fontsize = 10, borderaxespad=0.)
#         plt.show()  
#     plt.legend(bbox_to_anchor=(1.02, 1.12), loc=2, fontsize = 14, borderaxespad=0.)
#     plt.show()    
def main():
#     plotKeywordDistibution()
    dist()
#     keywordDistibution()
#     plotKeywordDistibution()
#     documents = getDocuments()
#     getDictionary(documents)
#     sys.stdout = Logger()
#     graphDiameter()
#     readGraph()
#     checkGraph()
#     (authors, authorsRev) = generateCitationList()
#     generateAuthorGraph(authors, authorsRev)
if __name__ == '__main__':
    main()