from numpy import sin as sin
from numpy import cos as cos
from numpy import pi as pi
class Setting:
    def __init__(self):
        self.cellWidth = 30
        self.alpha = pi/3
        self.dx=self.cellWidth*cos(self.alpha)
        self.dy=self.cellWidth*sin(self.alpha)
        self.color = [(99,9,0),(250,147,99),(195,204,215),(77,199,38),(23,63,63),(0,2,61)]
        self.cubeColor = 'black'
        self.boardSize = 16
        self.screenWidth = int((self.boardSize+2)*self.cellWidth+(self.boardSize+1)*self.dx)
        self.screenHeight = int((self.boardSize+4)*self.dy)
        self.inputFileName = 'Input.txt'
        self.delayTime = 100
        self.timeLimit = 20