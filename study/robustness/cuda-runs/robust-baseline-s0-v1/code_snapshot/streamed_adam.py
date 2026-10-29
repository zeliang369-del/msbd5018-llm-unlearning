"""FP32-master AdamW with CPU gradient accumulation and disk-backed states.

The arithmetic is AdamW, not a different unlearning algorithm. Disk files are
temporary optimizer storage. FP32 masters preserve sub-BF16-ULP updates.
"""
import math
import mmap
from pathlib import Path
import shutil
import numpy as np
import torch


def host_tensor(tensor):
    result=tensor.detach().cpu()
    if tensor.device.type=="mps":torch.mps.synchronize()
    return result.contiguous()


class StreamedAdamW(torch.optim.Optimizer):
    def __init__(self, groups, named_parameters, directory, lr=1e-5,
                 betas=(0.9, 0.999), eps=1e-8, chunk_elements=1_048_576,
                 reserve_bytes=2 * 2**30,capture_gradients=True,row_sparse_parameters=None,max_active_rows=16384):
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps, weight_decay=0.01))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.names = {id(p): name for name, p in named_parameters}
        self.capture_gradients=capture_gradients
        self.row_sparse=row_sparse_parameters or {}
        self.max_active_rows=max_active_rows
        self.active_rows={key:set() for key in self.row_sparse}
        self.gradient_coefficient=1.0
        self.offsets, self.parameters = {}, []
        size = 0
        for group in self.param_groups:
            for p in group["params"]:
                assert p.requires_grad and p.is_contiguous()
                self.offsets[id(p)] = size
                size += p.numel()
                self.parameters.append(p)
        required = size * 12
        physical_bound=required
        for p in self.parameters:
            if id(p) in self.row_sparse:
                width=self.row_sparse[id(p)]
                physical_bound-=p.numel()*8
                physical_bound+=min(p.numel()//width,max_active_rows)*width*8
        assert shutil.disk_usage(self.directory).free > physical_bound + reserve_bytes, "Insufficient scratch storage for FP32 master + two moments"
        self.maps = {key: np.memmap(self.directory / f"{key}.f32", dtype=np.float32,
                                  mode="w+", shape=(size,)) for key in ("master", "first", "second")}
        self.chunk = chunk_elements
        self.shadows, self.handles = {}, []
        self.steps = 0
        self.closed = False
        # New sparse files read as zero. Write only the FP32 master initially.
        for p in self.parameters:
            offset = self.offsets[id(p)]
            # Copy same dtype/device first: avoid MPS dtype conversion on views
            # with storage offsets, and keep host mappings separate from DMA.
            original_cpu=host_tensor(p).view(-1)
            for a in range(0, p.numel(), self.chunk):
                b = min(a + self.chunk, p.numel())
                self.maps["master"][offset+a:offset+b] = original_cpu[a:b].float().numpy()
            self._flush_drop(self.maps["master"],offset,p.numel())
            del original_cpu
            if capture_gradients:self.handles.append(p.register_post_accumulate_grad_hook(self._capture))
        self.maps["master"].flush()
        self.scratch_bytes = required
        self.scratch_physical_bound_bytes=physical_bound

    def _flush_drop(self,mapping,start,elements):
        first=(start*4//mmap.PAGESIZE)*mmap.PAGESIZE
        last=min((start+elements)*4,mapping.nbytes)
        mapping._mmap.flush(first,last-first)
        if hasattr(mapping._mmap,"madvise"):
            try: mapping._mmap.madvise(mmap.MADV_DONTNEED,first,last-first)
            except (OSError,ValueError): pass

    def _capture(self, parameter):
        gradient = parameter.grad
        if gradient is None:
            return
        key = id(parameter)
        if key not in self.shadows:
            self.shadows[key] = torch.zeros(parameter.numel(), dtype=torch.float32, device="cpu")
        shadow = self.shadows[key]
        flat = host_tensor(gradient).view(-1)
        for a in range(0, flat.numel(), self.chunk):
            b = min(a+self.chunk, flat.numel())
            shadow[a:b].add_(flat[a:b].float())
        parameter.grad = None
        if parameter.device.type == "mps" and torch.mps.driver_allocated_memory() > 6 * 2**30:
            torch.mps.empty_cache()

    def clip_grad_norm(self, max_norm, norm_type=2, **kwargs):
        assert norm_type == 2
        if any(p.device.type=="mps" for p in self.parameters):torch.mps.empty_cache()
        square = 0.0
        values=self.shadows.values() if self.capture_gradients else (host_tensor(p.grad).view(-1) for p in self.parameters if p.grad is not None)
        for shadow in values:
            for a in range(0,shadow.numel(),self.chunk):
                g=shadow[a:a+self.chunk].to(device="cpu",dtype=torch.float32).numpy()
                # Avoid low-accuracy vendor BLAS SDOT reductions over very
                # large FP32 arrays; optimizer tensors themselves stay FP32.
                square += float(np.sum(np.square(g,dtype=np.float64),dtype=np.float64))
        norm = math.sqrt(square)
        self.last_gradient_norm=norm
        assert math.isfinite(norm),"Non-finite gradient norm"
        coefficient = min(1.0, float(max_norm)/(norm+1e-6))
        if self.capture_gradients and coefficient < 1:
            for g in self.shadows.values():
                g.mul_(coefficient)
        self.gradient_coefficient=1.0 if self.capture_gradients else coefficient
        return torch.tensor(norm, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        assert not self.closed and closure is None
        self.steps += 1
        for group in self.param_groups:
            lr, decay = float(group["lr"]), float(group["weight_decay"])
            beta1, beta2 = group["betas"]
            correction1 = 1-beta1**self.steps
            correction2_sqrt = math.sqrt(1-beta2**self.steps)
            for p in group["params"]:
                shadow=self.shadows.get(id(p)) if self.capture_gradients else (None if p.grad is None else host_tensor(p.grad).view(-1))
                if shadow is None:
                    continue
                offset = self.offsets[id(p)]
                width=self.row_sparse.get(id(p))
                chunk=max(width,(self.chunk//width)*width) if width else self.chunk
                for a in range(0,p.numel(),chunk):
                    b=min(a+chunk,p.numel())
                    region = slice(offset+a, offset+b)
                    master, first, second = (self.maps[k][region] for k in ("master", "first", "second"))
                    gradient=shadow[a:b].to(device="cpu",dtype=torch.float32).numpy()
                    if not self.capture_gradients:gradient=gradient*np.float32(self.gradient_coefficient)
                    master *= np.float32(1-lr*decay)
                    if width:
                        g=gradient.reshape(-1,width);f=first.reshape(-1,width);s=second.reshape(-1,width);w=master.reshape(-1,width)
                        active=np.flatnonzero(np.any(g!=0,axis=1)|np.any(f!=0,axis=1)|np.any(s!=0,axis=1))
                        self.active_rows[id(p)].update((a//width+int(row)) for row in active)
                        assert len(self.active_rows[id(p)])<=self.max_active_rows,"Sparse state disk bound exceeded"
                        ff=f[active]*np.float32(beta1)+np.float32(1-beta1)*g[active]
                        ss=s[active]*np.float32(beta2)+np.float32(1-beta2)*g[active]*g[active]
                        denominator=np.sqrt(ss)/np.float32(correction2_sqrt)+np.float32(group["eps"])
                        w[active]-=np.float32(lr/correction1)*ff/denominator
                        f[active]=ff;s[active]=ss
                    else:
                        first *= np.float32(beta1)
                        first += np.float32(1-beta1) * gradient
                        second *= np.float32(beta2)
                        second += np.float32(1-beta2) * gradient * gradient
                        denominator = np.sqrt(second)/np.float32(correction2_sqrt)+np.float32(group["eps"])
                        master -= np.float32(lr/correction1) * first / denominator
                    if self.capture_gradients:shadow[a:b].zero_()
                # Stage a whole same-dtype host tensor; never release mmap pages
                # while a GPU conversion may still be reading their contents.
                host=torch.from_numpy(np.asarray(self.maps["master"][offset:offset+p.numel()])).to(dtype=p.dtype).reshape(p.shape)
                p.copy_(host,non_blocking=False)
                if p.device.type=="mps":torch.mps.synchronize()
                del host
                for mapping in self.maps.values():self._flush_drop(mapping,offset,p.numel())
                if p.device.type=="mps":torch.mps.empty_cache()
        self.gradient_coefficient=1.0
        return None

    def cleanup(self):
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.shadows.clear()
        for mapping in self.maps.values():
            mapping.flush()
            mapping._mmap.close()
        self.maps.clear()
        for key in ("master", "first", "second"):
            (self.directory / f"{key}.f32").unlink()
        self.directory.rmdir()
        self.closed = True
