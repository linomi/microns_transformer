import numpy as np 
def seq_generator(movie:np.array,running:np.array,trace:list,p,bsize:int,delay:int,shuffle = True):
   
    '''
    movie: numpy array with shape of (samples,h,w,c)
    running : locomotion data with shape of (samples)
    trace: fluorescence trace with shape of (samples,#neurons)
    p : list of indecies that used for given input domain i.e. natural movie
    bsize : batch size 
    delay: delay of input in compare to output 
    shuffle: used for training of model, for inference purpose turn it False


    return tupple : ([movie stimuli ,locmotion,trace past],trace)

    '''
    p = p[:len(p)-(len(p)%bsize)]
    p = p.reshape((len(p)//bsize,-1))
    shape = p.shape
    movie_shape = movie.shape[1:]
    trace_shape = trace.shape[-1]
    while True: 
        if shuffle:
            p = p.flatten() 
            np.random.shuffle(p)
            p = p.reshape(shape)
        for s in p: 
           t = trace[s]
           get_idx=np.array([[x for x in range(o-delay,o)] for o in s]).flatten()
           r = running[get_idx].reshape(bsize,delay)
           out = movie[get_idx].reshape(bsize,delay,movie_shape[0],movie_shape[1],movie_shape[2])
           feedback = trace[get_idx-1].reshape(bsize,delay,trace_shape)
           yield ([out,r,feedback],t)
