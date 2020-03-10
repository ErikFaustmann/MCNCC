def compute_cmc(score_mat):
    
    test_labels = [26, 8, 28, 37, 23, 5, 17, 27, 1, 15, 15, 8, 8, 30, 35, 6, 1, 32, 22, 1, 1, 19, 13, 20, 1, 7, 21, 36, 3, 12, 33, 9, 34, 38, 12, 11, 10, 14, 16, 29, 4, 24, 4, 2, 33, 3, 18, 31, 25, 25]

    true_mat = np.zeros((50, 38))

    for i in range(len(test_labels)):
        true_mat[i, test_labels[i]-1]= 1
    
    cmc = np.zeros(score_mat.shape[1], dtype='float64' )
    mx = np.zeros(score_mat.shape[0], dtype='float64')
    true_mat_est = np.zeros(score_mat.shape)
    est_loc =np.zeros(score_mat.shape[0])
    score_mat2 = score_mat

    for i in range(score_mat.shape[1]):
#________________________________________________________________________________ 
    #mx = np.zeros(score_mat.shape[0], dtype='float64')

        for w in range(score_mat.shape[0]):
            mx[w] = max(score_mat2[w])
    #print(mx)
#________________________________________________________________________________ 
    #true_mat_est = np.zeros(score_mat.shape)

        for e in range(score_mat.shape[0]):
            true_mat_est[e]  = np.equal(score_mat2[e], mx[e])

            est_loc[e] = list(true_mat_est[e]).index(1)
        if i == 0:
            with np.printoptions(threshold=np.inf):
                #print(true_mat_est)
                print(est_loc)

        true_mat_est = true_mat_est*1
    #print(i, ":", sum(true_mat_est))
#________________________________________________________________________________     
    
        if i == 0:
            cmc[i] = np.tensordot(true_mat, true_mat_est, axes=2)/score_mat.shape[0]
            #print("correct matches:", numpy.tensordot(true_mat, true_mat_est, axes=2))
        else:
            cmc[i] = (np.tensordot(true_mat, true_mat_est, axes=2)/score_mat.shape[0])+ cmc[i-1]
            #print("correct matches:", numpy.tensordot(true_mat, true_mat_est, axes=2))
#________________________________________________________________________________ 
        for g in range(score_mat.shape[0]):
            score_mat2[g][int(est_loc[g])] = -100000
            #print(est_loc[g])
            #print(score_mat2[g])

    return cmc