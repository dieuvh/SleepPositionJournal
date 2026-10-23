import unittest
import numpy as np
from utils.load_data import generate_data
class TestWindows(unittest.TestCase):
 def test_label_boundary(self):
  x=np.arange(60,dtype=np.float32).reshape(20,3)
  y=np.r_[np.zeros(8,int),np.ones(12,int)]
  w,z=generate_data(x,y,5,2)
  self.assertEqual(len(z),6)
  for a,label in zip(w,z):
   self.assertTrue(np.all(y[(a[:,0]/3).astype(int)]==label))
 def test_exact_length(self):
  w,z=generate_data(np.ones((5,3)),np.zeros(5),5,1)
  self.assertEqual(w.shape,(1,5,3))
 def test_short_runs(self):
  w,z=generate_data(np.ones((6,3)),np.r_[np.zeros(3),np.ones(3)],4,1)
  self.assertEqual(len(z),0)
if __name__=="__main__":unittest.main()
