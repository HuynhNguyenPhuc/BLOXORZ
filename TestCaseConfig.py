class TestCase:
    def __init__(self,filename):
        self.filename = filename
        self.cusor = 0

    def isNotEmpty(self):
        with open(self.filename,'r',encoding='utf-8') as f:
            f.seek(self.cusor)
            buf = f.readline().strip()
            return buf!=""
        
    def getTestCase(self):
        with open(self.filename,'r',encoding='utf-8') as f:
            f.seek(self.cusor)
            testCase=""
            buf = f.readline().strip()
            tc_name = buf
            self.cusor+=(len(buf)+2)
            buf = f.readline().strip()
            self.cusor+=(len(buf)+2)
            while buf!="END":
                testCase+=buf
                buf = f.readline().strip()
                self.cusor+=(len(buf)+2)
            return eval(testCase), tc_name