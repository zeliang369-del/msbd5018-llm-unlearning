"""CUDA-only adapters; frozen Mac sources are imported without edits."""
import functools
import mmap
import os

os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from omegaconf import OmegaConf
from full_core import BASE, UPSTREAM, PROMPT_DATE
from model import get_model
from streamed_adam import StreamedAdamW
from memory_gd import SequentialGradDiff


def configure(seed):
    from trainer.utils import seed_everything
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA device")


def load_cuda(path=BASE):
    cfg = OmegaConf.load(UPSTREAM / "configs/model/Llama-3.2-1B-Instruct.yaml")
    cfg.template_args.date_string = PROMPT_DATE
    cfg.model_args.pretrained_model_name_or_path = str(path)
    cfg.model_args.attn_implementation = "sdpa"
    cfg.model_args.torch_dtype = "bfloat16"
    cfg.model_args.local_files_only = True
    cfg.model_args.low_cpu_mem_usage = True
    cfg.tokenizer_args.pretrained_model_name_or_path = str(BASE)
    cfg.tokenizer_args.local_files_only = True
    model, tokenizer = get_model(cfg)
    model.config.use_cache = False
    model.to("cuda:0")
    tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, date_string=PROMPT_DATE)
    return model, tokenizer, cfg.template_args


class CudaGradDiff(SequentialGradDiff):
    """Existing sequential backward preserves the two original token-mean losses."""
    pass


class CudaStreamedAdamW(StreamedAdamW):
    def _flush_drop(self, mapping, start, elements):
        # Windows requires allocation-granularity-aligned mmap.flush offsets.
        granularity = mmap.ALLOCATIONGRANULARITY
        first = (start * 4 // granularity) * granularity
        last = min((start + elements) * 4, mapping.nbytes)
        mapping._mmap.flush(first, last - first)
        if hasattr(mapping._mmap, "madvise"):
            try:
                mapping._mmap.madvise(mmap.MADV_DONTNEED, first, last - first)
            except (OSError, ValueError):
                pass

    def step(self, closure=None):
        self.last_applied_lrs = [float(group["lr"]) for group in self.param_groups]
        return super().step(closure)
