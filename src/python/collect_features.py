import glob
import h5py
import os

def attributes(fp):
    t = fp.split('/')[-2]
    fp = fp.split('/')[-1]
    fp = fp.split('.')[0]
    try:
        m, ID, ps, l, ca = fp.split('/')[-1].split('_')
        ca = ca.split('.')[0].split('-')[1]
        l = l[1:]
        return t, m, ID, ps, l, ca
    except ValueError:
        m, ID, ps, l = fp.split('/')[-1].split('_')
        l = l[1:]
        return t, m, ID, ps, l
    
    

GTEx_dir = '/hps/nobackup/research/stegle/users/willj/GTEx'
jn = os.path.join
with h5py.File(jn(GTEx_dir,'data/h5py/collected_features.h5py'),'w') as f:
    for (i,fp) in enumerate(glob.glob(jn(GTEx_dir,'data/features/*/*'))):
        if i % 10 == 0:
            print (i) 
        res = attributes(fp)
        g = h5py.File(fp)
        features = g['features']
        
        group = f.create_group('/'+ '/'.join(res))
        group.create_dataset('features', data=features)
        g.close()

