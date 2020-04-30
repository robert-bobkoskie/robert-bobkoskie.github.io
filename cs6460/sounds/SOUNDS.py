
import numpy as np
import sys, time, math
import winsound, threading#,pydub

#import time, numpy, pygame.mixer, pygame.sndarray
#from scikits.samplerate import resample

def Check_Grid_Idx(grid, idx):
    is_valid = True

    try:
        grid[idx]
        if idx[0] < 0 or idx[1] < 0:
            is_valid = False
            #print '1 here', idx, grid[idx]
        elif np.isnan(grid[idx]):
            is_valid = False
        else:
            is_valid = True
            #print 'TRUE'
    except:
        is_valid = False
        #print '2 here', idx, grid[idx]

    return is_valid

def NUMPY():
    #winsound.MessageBeep()
    #winsound.Beep(32767, 5000) #Very High Pitch
    #winsound.Beep(300, 100) #Very Low Pitch

    #threading.Thread(target=winsound.Beep, args=(44, 2000)).start()
    threading.Thread(target=winsound.Beep, args=(240, 2000)).start()
    #threading.Thread(target=time.sleep(4)).start()
    threading.Thread(target=winsound.Beep, args=(1000, 2000)).start()
    #winsound.Beep(300, 200)
    #winsound.Beep(500, 200)
    #for x in range(8): winsound.Beep(300*(x+1), 500)

#############################
def main():
    #Py_Brain()
    NUMPY()

if __name__ == '__main__':
    main()
#############################
