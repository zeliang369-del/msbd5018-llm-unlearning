"""NPO-only dataset/collator adapters; GA/GD data flow is untouched."""
import torch
from torch.utils.data import Dataset
from data.collators import DataCollatorForSupervisedDataset
from memory_gd import pad_batch

class NPOForgetDataset(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        item = dict(self.dataset[index]); item["sample_id"] = int(index); return item

class NPOTracedPairs(Dataset):
    def __init__(self, forget, retain):
        self.forget, self.retain, self.trace = NPOForgetDataset(forget), retain, []
    def __len__(self): return len(self.forget)
    def __getitem__(self, index):
        retain_index = torch.randint(0, len(self.retain), (1,)).item()
        self.trace.append({"forget_index": index, "retain_index": retain_index})
        return {"forget": self.forget[index], "retain": self.retain[retain_index]}

class NPOTrainingCollator(DataCollatorForSupervisedDataset):
    def __init__(self, tokenizer, length):
        super().__init__(tokenizer)
        self.length = length

    def __call__(self, instances):
        if "input_ids" not in instances[0]:
            return {key: self(self.get_instances_from_key(instances, key)) for key in instances[0]}
        sample_ids = [instance.pop("sample_id") for instance in instances] if "sample_id" in instances[0] else None
        batch = super().__call__(instances)
        batch = pad_batch(batch, self.length, self.tokenizer.pad_token_id)
        if sample_ids is not None: batch["sample_id"] = torch.tensor(sample_ids, dtype=torch.long)
        return batch
