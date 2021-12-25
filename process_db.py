ls = open("models/database_org.txt",encoding="utf-8").readlines()
n = len(ls)
lsa= []
for i in range(0,len(ls)):
    ls[i]=ls[i].strip()
    if(len(ls[i])<150):
        lsa.append(ls[i])
tf = open("models/database.txt","a+",encoding="utf-8")
ln = len(lsa)
for i in range(0,ln):
    tf.write(lsa[i])
    tf.write("\n")
