"""Validate streamed FP32-master arithmetic against standard torch AdamW."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from streamed_adam import StreamedAdamW


def main():
    torch.manual_seed(17)
    initial = torch.randn(257).to(torch.bfloat16)
    low_precision = torch.nn.Parameter(initial.clone())
    reference = torch.nn.Parameter(initial.float().clone())
    ordinary = torch.optim.AdamW([{"params": [reference], "weight_decay": 0.01}], lr=1e-5, eps=1e-8, foreach=False)
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        streamed = StreamedAdamW([{"params": [low_precision], "weight_decay": 0.01}],
                                [("test", low_precision)], Path(directory)/"state", chunk_elements=37, reserve_bytes=0)
        errors = []
        for step in range(20):
            rate = 1e-5*(20-step)/20
            streamed.param_groups[0]["lr"] = ordinary.param_groups[0]["lr"] = rate
            gradient = torch.randn(257).to(torch.bfloat16).float()
            # Hook captures a real BF16 backward gradient, rather than a fake state.
            (low_precision.float()*gradient).sum().backward()
            reference.grad = gradient.clone()
            streamed.clip_grad_norm(1.0)
            torch.nn.utils.clip_grad_norm_([reference],1.0)
            streamed.step(); ordinary.step(); ordinary.zero_grad()
            master = torch.from_numpy(np.array(streamed.maps["master"]))
            error = float((master-reference.detach()).abs().max())
            errors.append(error)
            assert error < 4e-7
            assert torch.equal(low_precision.detach(), reference.detach().to(torch.bfloat16))
        assert float((master-initial.float()).abs().max()) > 0
        report = {"status": "passed", "updates": 20, "FP32_master_max_absolute_error": max(errors),
                  "BF16_parameter_matches_rounded_standard_AdamW": True,
                  "purpose": "Verify sub-BF16-ULP accumulation and standard AdamW equivalence, not just finite loss"}
        streamed.cleanup()
    proof = Path(__file__).resolve().parent / "reproduction_full/streamed-adam-verification.json"
    proof.parent.mkdir(parents=True, exist_ok=True)
    proof.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))

if __name__ == "__main__": main()
