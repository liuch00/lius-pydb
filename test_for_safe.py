import os
path = ['./backend',
        './Exceptions',
        './FileSystem',
        './IndexSystem',
        './ManageSystem',
        './MetaSystem',
         # './SQL_Parser',
        './RecordSystem']
for direc in path:

    for filename in os.listdir(direc):
        fullname = os.path.join(direc, filename)
        if os.path.isfile(fullname):
            f1 = open(fullname, encoding='gb18030', errors='ignore')
            for i in f1:
                with open("all1.py", encoding='utf-8', mode='a+')as f2:
                    f2.write(i)
                    f2.close()
            f1.close()