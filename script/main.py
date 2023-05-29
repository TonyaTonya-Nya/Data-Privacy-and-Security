import r2pipe, os, sys, time, csv
import pandas as pd
import numpy as np




def main():
    
    cmd = 'mkdir ./Redefine/ ./DetectionUserDefined/ ./cfg_output/'
    os.system(cmd)
    
    ### CG_extract 會從bin 提取fcg 到cfg_output  ####
    cmd = 'python CG_extract.py'
    os.system(cmd)
    
    ### 會從bin中提取funcion name 到 DetectionUserDefined/ 中  ###
    cmd = 'python AnalysisUserDefined.py'
    os.system(cmd)

    ### 檢查重複 ###
    cmd = 'fdupes -r ./DetectionUserDefined/ > ./fdupes.txt'
    os.system(cmd)
    
    ### 將 fdupes.txt 與 cfg_output 裡的graph 結合 重新編輯  ###
    cmd = 'python CreateRenameGraph.py'
    os.system(cmd)

    ### 產出 feature  ###
    cmd = 'python graph2vec+.py --input-path ./Redefine/ --output-path ./test.csv'
    os.system(cmd)
    
    

if __name__=='__main__':
    main()

