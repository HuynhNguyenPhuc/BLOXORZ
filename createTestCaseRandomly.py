import random as rd
with open('Input.txt', mode='a',encoding='utf-8') as f:
    f.writelines("TESTCASE\n")
    board = {}
    for i in range (0,16):
        for j in range(0,16):
            board[(i,j)]=rd.randint(rd.randint(0,1),2)
    board[(12,11)]=3
    board[(2,2)]=2
    startPos = [(2,2),(2,2)]
    button = {}
    for i in range(0,6):
        pos =(rd.randint(2,15),rd.randint(2,15))
        while board[pos]==0 or board[pos]==3:
            pos =(rd.randint(2,15),rd.randint(2,15))
        active = []
        kind = rd.randint(1,3)
        if kind==3:
            active.append((rd.randint(0,15),rd.randint(0,15)))
            active.append((rd.randint(0,15),rd.randint(0,15)))
        else:
            for j in range(0, rd.randint(0,10)):
                active.append((rd.randint(0,15),rd.randint(0,15)))
        button[pos] = [kind,active]
    for i in range(0,6):
        pos =(rd.randint(2,15),rd.randint(2,15))
        while board[pos]==0 or board[pos]==3:
            pos =(rd.randint(2,15),rd.randint(2,15))
        active = []
        kind = rd.randint(3,3)
        if kind==3:
            active.append((rd.randint(0,15),rd.randint(0,15)))
            active.append((rd.randint(0,15),rd.randint(0,15)))
        else:
            for j in range(0, rd.randint(0,4)):
                active.append((rd.randint(0,15),rd.randint(0,15)))
        button[pos] = [kind,active]
    tc = [board,startPos,button]
    f.writelines(str(tc)+'\n')
    f.writelines("END\n")