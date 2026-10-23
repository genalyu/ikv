"""Length grouping within effective updates without omitting any sample."""
from transformers.trainer_pt_utils import LengthGroupedSampler,get_length_grouped_indices
class FullHistoryLengthSampler(LengthGroupedSampler):
 def __init__(self,*args,mega_batch_mult=None,**kwargs):
  super().__init__(*args,**kwargs);self.mega_batch_mult=mega_batch_mult
  self.seed=self.generator.initial_seed();self.epoch=0
 def set_epoch(self,epoch):
  self.epoch=int(epoch)
 def __iter__(self):
  self.generator.manual_seed(self.seed+self.epoch)
  return iter(get_length_grouped_indices(self.lengths,self.batch_size,mega_batch_mult=self.mega_batch_mult,generator=self.generator))
