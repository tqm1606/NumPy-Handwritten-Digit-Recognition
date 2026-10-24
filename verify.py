"""Kiểm tra split, gradient backprop, lưu/nạp và các khung trống."""
import tempfile
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from train_from_scratch import NeuralNetwork, stratified_split
from camera_digit_recognizer import TrainedNetwork
from preprocessing import preprocess_roi

def main():
    y=load_digits().target
    splits=stratified_split(y)
    assert len(np.unique(np.concatenate(splits)))==len(y)
    assert sum(map(len,splits))==len(y)
    rng=np.random.default_rng(8)
    m=NeuralNetwork(sizes=(4,5,3),dropout=0,l2=0)
    m.W=[w.astype(float) for w in m.W]; m.b=[b.astype(float) for b in m.b]
    X=rng.random((6,4)); labels=np.array([0,1,2,0,1,2]); eps=1e-5
    def loss():
        p=m.predict_proba(X)
        return -np.log(p[np.arange(6),labels]).mean()
    numeric=[]
    for layer, i, j in [(0,0,0),(0,2,3),(1,1,2),(1,4,0)]:
        old=m.W[layer][i,j]
        m.W[layer][i,j]=old+eps; plus=loss()
        m.W[layer][i,j]=old-eps; minus=loss()
        m.W[layer][i,j]=old
        numeric.append((layer,i,j,(plus-minus)/(2*eps)))
    m.train_batch(X,labels,0,rng,label_smoothing=0)
    for layer,i,j,g in numeric:
        assert np.isclose(m.mW[layer][i,j]/.1,g,rtol=1e-4,atol=1e-6)
    m=NeuralNetwork()
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'model.npz'; m.save(path)
        loaded=TrainedNetwork(path); x=rng.random((1,64)).astype(np.float32)
        assert np.allclose(m.predict_proba(x)[0],loaded.predict_proba(x[0]),atol=1e-6)
    for value in (0,127,255):
        assert preprocess_roi(np.full((112,112),value,np.uint8))[0] is None
    print('PASS: split không trùng; gradient số khớp backprop; save/load khớp; khung trống bị từ chối.')
if __name__=='__main__':
    main()

