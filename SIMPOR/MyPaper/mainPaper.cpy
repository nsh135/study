\clipboard {abstract}{ This work explores how class imbalanced data affects deep learning and proposes a data balancing technique for mitigation by generating more synthetic data for the minority class. In contrast to random-based oversampling techniques, our approach prioritizes balancing the most informative region by finding high entropy samples. This approach is opportunistic and challenging because well-placed synthetic data points can boost machine learning algorithms' accuracy and efficiency, whereas poorly-placed ones can cause a higher misclassification rate. In this study, we present an algorithm for maximizing the probability of generating a synthetic sample in the correct region of its class by placing it toward maximizing the class posterior ratio. In addition, to preserve data topology, synthetic data are closely generated within each minority sample neighborhood. Experimental results on forty-two datasets show that our technique significantly outperforms all compared state-of-the-art ones in terms of boosting deep-learning performance. It has been shown that a deep learning model can achieve up to 12\% higher F1 score when trained with a data set augmented by the proposed technique than with the current state-of-the-art techniques. It also archives the highest winning times in F1-score and AUC over 41 datasets compared to others. }
\clipboard {parameterSelection}{ We randomly split the datasets into two parts, 80\% for training and 20\% for testing. Reported testing results for each dataset are the averages of 5 experimental trials. For \Methodname {} to find optima of the function in Equation \ref {prob:optimazation}, we use a gradient ascent rate of 0.00001 and the iteration of 300. The architecture detail of the evaluation classifier and technique parameters are described in Table \ref {tab:model_setting}. }
