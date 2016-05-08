#! /usr/bin/env python
# coding=utf-8

'''
Created on 2015年10月16日

@author: yanruibo
'''


import random

import cv2

if __name__ == '__main__':
    '''
    binarySquence = [1, 1, 1, 1, 0, 0, 0, 1]
    compareValues = []
    for i in range(8):
        sum = 0
        pos = i
        for j in range(8):
            sum += binarySquence[pos % 8] * pow(2, 7 - j)
            # print "pos",pos
            # print "number",binarySquence[pos%8]
            pos = pos + 1
        print sum
    '''
    '''
    sortedValues = []
    k = 0
    counter = 0
    for i in range(len(sortedValues)):
        temp = sortedValues[k]
        if(temp == sortedValues[i]):
            counter = counter + 1
        else:
            k = i
            counter = 0
    '''
    '''
    ans = get36Values()
    print ans
    print len(ans)
    '''
    mat = cv2.imread('../Database_PR_01/neg (1).png')
    print "rgb image shape",mat.shape
    print mat
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    print "gray image shape",gray.shape
    print gray
    
    x = LBP.LBP(gray)
    print "after LBP"
    print x

    # cv2.imshow("imgrgb",mat)
    # cv2.imshow("gray",gray)
    # cv2.waitKey()
    # print gray.shape
    # print gray
    '''
    train_neg_indexes=[49, 583, 685, 835, 770, 678, 269, 290, 442, 146, 780, 226, 368, 422, 363, 94, 660, 748, 651, 256, 963, 694, 76, 72, 446, 656, 670, 710, 822, 393, 738, 714, 716
                              , 23, 7, 232, 482, 323, 531, 221, 447, 458, 259, 314, 862, 831, 867, 975, 24, 324, 28, 276, 371, 720, 212, 966, 405, 479, 737, 287, 533, 630, 750, 174, 496, 618
                              , 645, 0, 755, 717, 260, 861, 331, 935, 117, 285, 121, 661, 807, 471, 844, 365, 
                              701, 46, 893, 897, 220, 846, 930, 885, 864, 441, 230, 10, 990, 937, 602, 26, 92,101]
    train_pos_indexes=[760, 743, 292, 496, 461, 751, 926, 512, 521, 929, 454, 918, 103, 995, 413, 279,
                                     361, 915, 714, 179, 730, 961, 813, 107, 816, 135, 253, 777, 445, 697, 593, 468,
                                     73, 669, 443, 640, 169, 594, 647, 891, 811, 772, 317, 324, 790, 686, 123, 294, 
                                     654, 672, 737, 529, 282, 774, 10, 902, 806, 17, 45, 653, 878, 210, 505, 582, 485, 767, 464, 344, 950, 289, 415, 525, 636, 183, 451, 387, 204, 473, 859, 855, 110
                                     , 823, 450, 759, 221, 945, 453, 228, 469, 928, 230, 218, 384, 499, 463, 753, 39,
                                     754, 121, 99]
    
    np.savetxt("train_neg_indexes_100.txt", train_neg_indexes)
    np.savetxt("train_pos_indexes_100.txt", train_pos_indexes)
    '''
   
    
            