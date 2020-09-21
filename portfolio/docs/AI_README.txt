Note:	Agent.py in the zip is the original submission, and does not access
	'problem.checkAnswer' to verify, and print the result out to the
	command line. The Agent.py in this folder does verify, and print.

	Agent_FIXES.py in the zip file is modified to call 'problem.checkAnswer'.


	'''
        if self.debugL1:
            myAnswer = max(normMaxScore)
            print normMaxScore
            for x in range(len(normMaxScore)):
                if normMaxScore[x] == myAnswer:
                    print 'MY ANS:', (x + 1)
        '''

        if self.debugL1:
            myAnswer = max(normMaxScore)
            answer = problem.checkAnswer(normMaxScore)
            print normMaxScore
            for x in range(len(normMaxScore)):
                if normMaxScore[x] == myAnswer:
                    if ((x + 1) == answer):
                        print 'CORRECT', 'MY ANS:', (x + 1), 'ANS:', answer
                    else:
                        print 'ERROR', 'MY ANS:', (x + 1), 'ANS:', answer
