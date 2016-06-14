import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        n_words=params['n-words'][0],
                                        n_words_src=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0],
                                        maxlen=50,
                                        batch_size=32,
                                        valid_batch_size=32,
					datasets=['../data/train.data','../data/train.data'],
					valid_datasets=['../data/valid.data','../data/valid.data'],
					dictionaries=['../data/train.data.pkl','../data/train.data.pkl'],
                                        validFreq=5000,
                                        dispFreq=100,
                                        saveFreq=5000,
                                        sampleFreq=1000,
                                        use_dropout=params['use-dropout'][0],
                                        overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_session1.npz'],
        'dim_word': [100],
        'dim': [200],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'use-dropout': [True],
        'learning-rate': [0.0001],
        'reload': [False]})


