 Grow_tree + CCP_prune:  Be careful not to delete the train.csv and test.csv file in the folder, otherwise you can't run the program.
                                            
		         Delete the annotation of line 383, you will get the number of leaf nodes of the final rivised tree.
                                          
		         Delete the annotation of line 384, you will get the final rivised tree.
                                           	
		         Delete the annotation of line 347 and line 459, you will see the number of leaf nodes after each pruning.

Grow_tree + PEP / REP_prune: The program is used to report the accuracy of predicting the quality of red wines.

			 Run the program, and it will automatically build a classification tree to predict whether the quality score of the red wine is above 6 or not. Then it will display the accuracy of the prediction before tree pruning and after tree pruning.

			 If you want the program to show the accuracy in predicting the quality of other red wines, put the file containing the data(which must be in the same format as the test.csv file in the folder) into the folder and rename it to test.csv.

			 Be careful not to delete the train.csv and test.csv file in the folder, otherwise the program may not be able to build the classification tree and do the prediction.