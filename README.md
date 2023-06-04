# merge
Very simple Extension for oobabooga web ui to merge Base model with Lora

(can merge only half float HF with LORA, so no 4-nit, or 8 bit)

adds Merge tab, allows loading HF model and merging with LORA


## Issue:
Because ooba already uses some GPU, you are more likely to fail due CUDA error than if you run this outside ooba

So it doesn't make much sense.
But it can surely merge smaller models.

device map is set as:
device_map={'': 0}


TODO:
- See if the GPU can be somehow optimized (??)
- merge 4 bit GPTQ with LORA (no idea how)
- some feedback in interface

