
'''
asaraf@google.com
 
Implement Conway's Game of Life. 
For a 2D m by n rectangular grid, each cell can be on or off. 
Each cell updates based on its eight neighbors. 
If 2 neighbors are on, the state remains the same. 
If 3 neighbors are on, the state becomes on. 
Otherwise the state becomes off. 
 
a b c d
e f g h
i j k l
'''

def Conway(matrix):
    # Use a list comprehension
    row = len(matrix) - 1
    col = len(matrix[0]) - 1
    update_mat = [ [0.0 for col in range(len(matrix[0]))] for row in range(len(matrix)) ]
    #print update_mat

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            count_on = 0
            count_off = 0
            cur_state = matrix[i][j]

            #Cell 1
            if j-1 >= 0:
                if matrix[i][j-1]:
                    count_on +=1
                else:
                    count_off +=1
                print ' ---', cur_state, count_on, count_off
            else:
                pass

            #Cell 2
            if j+1 <= col:
                if matrix[i][j+1]:
                    count_on +=1
                else:
                    count_off +=1
                print ' ---', cur_state, count_on, count_off
            else:
                pass

            #Cell 3
            if i-1 >= 0:
                if matrix[i-1][j]:
                    count_on +=1
                else:
                    count_off +=1
                print ' ---', cur_state, count_on, count_off
            else:
                pass

            #Cell 4
            if i+1 <= row:
                if matrix[i+1][j]:
                    count_on +=1
                else:
                    count_off +=1
                print ' ---', cur_state, count_on, count_off
            else:
                pass

            #Cell 5
            if (i-1 >= 0 and j-1 >= 0):
                if matrix[i-1][j-1]:
                    count_on +=1
                else:
                    count_off +=1
                print ' ---', cur_state, count_on, count_off
            else:
                pass

            #Cell 6
            if i-1 >= 0 and j+1 <= col:
                if matrix[i-1][j+1]:
                    count_on +=1
                else:
                    count_off +=1
                print ' ---', cur_state, count_on, count_off
            else:
                pass

            #Cell 7
            if j-1 >= 0 and i+1 <= row:
                if matrix[i+1][j-1]:
                    count_on +=1
                else:
                    count_off +=1
                print ' ---', cur_state, count_on, count_off
            else:
                pass

            #Cell 8
            if i+1 <= row and j+1 <= col:
                if matrix[i+1][j+1]:
                    count_on +=1
                else:
                    count_off +=1
                print ' ---', cur_state, count_on, count_off
            else:
                pass

            print cur_state, ':',  count_on, count_off, '\n'


            #If 2 neighbors are on, the state remains the same. 
            #If 3 neighbors are on, the state becomes on. 
            #Otherwise the state becomes off.
            if count_on == 2:
                update_mat[i][j] = cur_state
            elif count_on == 3:
                update_mat[i][j] = 1
            else:
                update_mat[i][j] = 0

    return update_mat
                

my_mat = [ [1, 2, 3],
           [4, 5 ,6],
           [7, 8, 9] ]

my_mat = [ [0, 0, 1],
           [0, 1 ,0],
           [1, 0, 0] ]

my_mat = [ [1, 0, 1],
           [0, 1 ,0],
           [1, 0, 1] ]

my_mat = [ [1, 0, 1, 0],
           [0, 1 ,0, 0],
           [1, 0, 1, 1] ]
'''
my_mat = [ [0, 0, 1],
           [0, 1 ,0],
           [1, 0, 0],
           [1, 1, 1] ]
'''
update_mat = Conway(my_mat)
print update_mat
for x in update_mat: print [ elem for elem in x ]

