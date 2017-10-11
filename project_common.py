import pandas as pd

class ProjectCommon:
    @staticmethod
    def clean_data(df):
        
        #print 'cleaning data: please add to this function'
        df.iloc[1] = 1 
        return df
        

        
