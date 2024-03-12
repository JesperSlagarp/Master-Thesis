import pandas as pd
import numpy as np
import os
from params import QUIPU_DATA_FOLDER,QUIPU_LEN_CUT,QUIPU_STD_FILL_DEFAULT

def normaliseLength(trace, length = QUIPU_LEN_CUT, trim = 0, std_default=QUIPU_STD_FILL_DEFAULT): ##Paramters given in quipus code
    """
    Normalizes the length of the trace and trims the front 
    
    :param length: length to fit the trace into (default: 600)
    :param trim: how many points to drop in front of the trace (default: 0)
    :return: trace of length 'length' 
    """
    if len(trace) >= length + trim:
        return trace[trim : length+trim]
    else:
        return np.append(
            trace[trim:],
            np.random.normal(0, std_default, length - len(trace[trim:]))
        )    

##Loads all the datasets as a df with traces normalized and number of nanopore and code.
def allDataset_loader(data_folder,path_dataset_preprocessed="../../data/dataset.hdf5",cut=True):
    df_name='datasetQuipu' if cut else 'datasetQuipuUncut'
    if os.path.exists(path_dataset_preprocessed):
        allDatasets=pd.read_hdf(path_dataset_preprocessed, df_name)  
    else:
        #Datasets selected for Quipu training in the original script
        dataset =         pd.concat([ 
            pd.read_hdf(data_folder+"dataset_part1.hdf5"),
            pd.read_hdf(data_folder+"dataset_part2.hdf5")
        ])
        #datasetTestEven = pd.read_hdf(data_folder+"datasetTestEven.hdf5")
        #datasetTestOdd =  pd.read_hdf(data_folder+"datasetTestOdd.hdf5")
        #datasetTestMix =  pd.read_hdf(data_folder+"datasetTestMix.hdf5")
        datasetWithAntibodies =  pd.concat([ 
            pd.read_hdf(data_folder+"datasetWithAntibodies_part1.hdf5"),
            pd.read_hdf(data_folder+"datasetWithAntibodies_part2.hdf5")
        ])
        datasetExtra =    pd.read_hdf(data_folder+"datasetExtra.hdf5")
        
        allDatasets = pd.concat([dataset , datasetExtra, datasetWithAntibodies],ignore_index = True)
        allDatasets = allDatasets[allDatasets.Filter] # clear bad points
    
        #Here I should normalize the traces
        traces = allDatasets.trace
        traces_uniform = traces.apply(lambda x: normaliseLength(x))
        traces_normalised_unif =  - traces_uniform / allDatasets.UnfoldedLevel 
        traces_normalised = - traces / allDatasets.UnfoldedLevel;
        allDatasets.trace = traces_normalised_unif;
        quipu_ds=allDatasets[ ["barcode", "nanopore","trace"]]; #Keep the only information that we will use
        quipu_ds.to_hdf(path_dataset_preprocessed, 'datasetQuipu');
        allDatasets.trace = traces_normalised;
        uncut_ds=allDatasets[ ["barcode", "nanopore","trace"]]; #Keep the only information that we will use
        uncut_ds.to_hdf(path_dataset_preprocessed, mode='a', key='datasetQuipuUncut');
        allDatasets=quipu_ds if cut else uncut_ds;
    return allDatasets;

def load_unfolded_df(data_folder="../../ext/QuipuData/"):
    dataset =         pd.concat([ 
        pd.read_hdf(data_folder+"dataset_part1.hdf5"),
        pd.read_hdf(data_folder+"dataset_part2.hdf5")
    ])
    datasetWithAntibodies =  pd.concat([ 
        pd.read_hdf(data_folder+"datasetWithAntibodies_part1.hdf5"),
        pd.read_hdf(data_folder+"datasetWithAntibodies_part2.hdf5")
    ])
    datasetExtra =    pd.read_hdf(data_folder+"datasetExtra.hdf5")
    
    allDatasets = pd.concat([dataset , datasetExtra, datasetWithAntibodies],ignore_index = True)
    allDatasets = allDatasets[allDatasets.Filter] # clear bad points
    traces = allDatasets.trace
    traces_uniform = traces.apply(lambda x: normaliseLength(x))
    traces_normalised_unif =  - traces_uniform / allDatasets.UnfoldedLevel 
    allDatasets.trace = traces_normalised_unif;
    df_unfolded=allDatasets[~allDatasets["nbell_barcode"].isnull()]
    df_unfolded=df_unfolded[df_unfolded["nbell_barcode"]<8]
    return df_unfolded

def create_random_tuples(barcode,code_nanopores,meas_counts,min_perc,max_perc):
    perc_ds=np.asarray(meas_counts)/np.sum(meas_counts)*100;
    tuples_test_list=[];
    if np.all(perc_ds>max_perc): #If already picking one nanopore has a higher percentage than the max, picks the smallest one
        tuples_test_list.append( (barcode,code_nanopores[np.argmin(perc_ds)]) )
    else:
        to_select=np.where(perc_ds<max_perc)[0][:] #From the ones with low percentage
        n_bits=len(to_select) # n_bits to show all possible configurations 
        n_combinations_total=2**len(to_select);
        comb_array=np.zeros((n_combinations_total,n_bits)).astype(bool);
        comb_perc_acc=np.zeros((n_combinations_total,));
        for i in range(n_combinations_total): #Express all the combinations in binary selector
            format_str='{0:0'+str(n_bits)+ 'b}'
            list_chars=list(format_str.format(i))
            comb_array[i,:]=[i=='1' for i in list_chars];
            idxs_picked=to_select[comb_array[i,:]];
            comb_perc_acc[i]=np.sum(perc_ds[idxs_picked]); #The percentage of data used in dataset if configuration is picked
        comb_array_within_range=comb_array[np.logical_and(comb_perc_acc>min_perc,comb_perc_acc<max_perc),:]; # keeps only combinations that satisfy the percentage range of the test dataset
        selected_comb=np.random.choice(len(comb_array_within_range), 1)[0]; #Selects a random combination
        for i in range(len(to_select)): #Pushing indexes
            if comb_array_within_range[selected_comb,i]: 
                code_index=to_select[i]
                tuples_test_list.append( (barcode,code_nanopores[code_index]) )
    return tuples_test_list;
    
#Divides the dataset in test and train, generating random test sets in a way that the test dataset is 4-15% of the data and it is only from nanopores that are not present on the train dataset.
def dataset_split(allDatasets,min_perc=4,max_perc=15): ## 4% because 010 only can pick 4%.
    barcodes=np.unique(allDatasets["barcode"].to_numpy());
    
    testSetIndex=[];
    for barcode in barcodes:
        code_ds=allDatasets[allDatasets["barcode"]==barcode]
        code_nanopores=np.unique(code_ds["nanopore"].to_numpy());
        meas_counts= [ np.sum(code_ds["nanopore"]==i) for i in code_nanopores]
        tuples_test=create_random_tuples(barcode,code_nanopores,meas_counts,min_perc,max_perc)
        for i in tuples_test:
            testSetIndex.append(i)

    
    testSetSelection = allDatasets[["barcode", "nanopore"]]\
                            .apply(tuple, axis = 1)\
                            .isin(testSetIndex)
    
    testSet = allDatasets[ testSetSelection ]
    trainSet = allDatasets[ ~ testSetSelection ]
    
    return trainSet,testSet

def show_porcentages(trainSet,testSet):
    Y_train_barcode = np.vstack( trainSet.barcode.values )
    Y_test_barcode = np.vstack( testSet.barcode.values )
    labels = np.array(['000', '001', '010', '011', '100', '101', '110', '111'])
    samples_dist=pd.DataFrame({
    "Train": pd.Series(Y_train_barcode.flatten()).value_counts() ,
    "Test": pd.Series(Y_test_barcode.flatten()).value_counts() }, index = labels) 
    samples_perc=samples_dist.copy();
    for i in range(len(samples_dist)):
        samples_perc.iloc[i,0]=100*samples_dist.iloc[i,0]/(samples_dist.iloc[i,0]+samples_dist.iloc[i,1]);
        samples_perc.iloc[i,1]=100*samples_dist.iloc[i,1]/(samples_dist.iloc[i,0]+samples_dist.iloc[i,1]);
    print(samples_dist)
    print(samples_perc)
    return samples_perc;
 
def show_partition_nanopores(ds):
    aux=ds[ ["barcode", "nanopore"]]; 
    aux=ds.groupby(aux.columns.tolist(),as_index=False).size();
    return aux;
    

if __name__ == "__main__":
    data_folder="../ext/QuipuData/";
    allDatasets=allDataset_loader(data_folder)
    trainSet,testSet=dataset_split(allDatasets)
    samples_perc=show_porcentages(trainSet,testSet)
    
    print(show_partition_nanopores(trainSet))
    print(show_partition_nanopores(testSet))

