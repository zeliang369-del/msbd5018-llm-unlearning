"""Check actual device transfers and zero-rate AdamW parameter invariance."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from streamed_adam import StreamedAdamW


def main():
    assert torch.backends.mps.is_available();torch.manual_seed(31)
    original=torch.randn(1027,257).to(torch.bfloat16)
    parameter=torch.nn.Parameter(original.to("mps"))
    pattern=torch.randn_like(original).to(torch.bfloat16)
    (parameter.float()*pattern.float().to("mps")).sum().backward()
    staged=parameter.grad.detach().cpu().float().view(-1)
    errors=[]
    for a in range(0,parameter.numel(),16381):
        combined=parameter.grad.detach().view(-1)[a:a+16381].to(device="cpu",dtype=torch.float32)
        errors.append(float((combined-staged[a:a+16381]).abs().max()))
    assert torch.equal(staged.view_as(pattern),pattern.float())
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        optimizer=StreamedAdamW([parameter],[("mps_test",parameter)],Path(directory)/"state",capture_gradients=False,
                               chunk_elements=16381,reserve_bytes=0)
        norm=optimizer.clip_grad_norm(1)
        expected_norm=pattern.float().norm()
        actual=parameter.grad.detach().cpu().float()
        torch.mps.synchronize()
        expected64=float(pattern.float().double().square().sum().sqrt())
        print(json.dumps({"computed_norm":float(norm),"expected_norm":float(expected_norm),"expected64":expected64,
                          "gradient_after_initialization_matches":torch.equal(actual,pattern.float()),"gradient_max_error":float((actual-pattern.float()).abs().max()),
                          "view_transfer_max_error":max(errors)}),flush=True)
        assert abs(float(norm)-expected64)/expected64<1e-6
        optimizer.param_groups[0]["lr"]=0
        optimizer.step();torch.mps.synchronize()
        assert torch.equal(parameter.detach().cpu().view(torch.uint16),original.view(torch.uint16)),"Zero learning rate changed BF16 bits"
        master=torch.from_numpy(np.array(optimizer.maps["master"])).view_as(original)
        assert torch.equal(master,original.float())
        optimizer.cleanup()
    report={"status":"passed","gradient_matches_expected":True,"combined_view_transfer_max_error":max(errors),
            "zero_learning_rate_preserves_all_parameter_bits":True,"staged_transfer_norm":float(norm)}
    Path('reproduction_full/mps-transfer-verification.json').write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))


if __name__=="__main__":main()
