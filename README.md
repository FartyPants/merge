# merge
Extension for oobabooga web ui to merge Base with Lora

Merge extension for oobabooga webui

adds Merge tab, allows loading HF model and merging with LORA


## Issue:
Because ooba already uses some GPU, you are more likely to fail due CUDA error than if you run this outside ooba
So it doesn't make much sense.
But it can surely merge smaller models.

device map is set as:
device_map={'': 0}


