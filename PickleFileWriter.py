import os
import csv
import pickle
from math import ceil

def scanfolder():
    s = 0
    b = 0
    a = 0
    c = 0
    Action = ''
    hold = []
    for path, dirs, files in os.walk('/media/austin/AustinEXT/CSV'):
        #print(path)
        for f in files:
            if f.endswith('.csv'):
                temp_array = []
                out = []
                faa_file_path = os.path.join(path,f)
                print(faa_file_path)
                f = open(faa_file_path, 'r')
                readCSV = csv.reader(f)

                for row in readCSV:
                    temp_array.append(row)
                    #print(i)
                names = faa_file_path.split("/")
                Actionold = Action
                Subject = str(names[-3])
                Action = str(names[-2])
                out.append(temp_array)
                out.append(Subject)
                out.append(Action)
                #arr = str(out)
                if Actionold == Action:
                    hold.append(out)
                    a += 1
                elif Actionold != Action and a != 0:
                    a = 0 #reset counter
                    test = int(ceil(.2 * a))  #first 20% of the array/data from single action
                    eva = 2 * test	#next 20%

                    testout = [hold[i][:] for i in range(0,test)]	#test data created from the first 20%
                    evalout = [hold[i][:] for i in range(test,eva)]	#eval data created from the next 20%
                    trainout = [hold[i][:] for i in range(eva, a)]	#train data created from the final 60%

                    hold = []   #empty holding array asap to free up space

                    trainfile = open("{}_train.p".format(Subject),"wb")
                    testfile =  open("{}_test.p".format(Subject),"wb")
                    evalfile =  open("{}_eval.p".format(Subject),"wb")

                    for i in testout:
                         pickle.dump(i, testfile, protocol= pickle.HIGHEST_PROTOCOL)
                    testout = []

                    for i in evalout:
                         pickle.dump(i, evalfile, protocol= pickle.HIGHEST_PROTOCOL)
                    evalout = []

                    for i in trainout:
                         pickle.dump(i, trainfile, protocol=pickle.HIGHEST_PROTOCOL)
                    trainout = []

                    trainfile.close()
                    testfile.close()
                    evalfile.close()

scanfolder()
