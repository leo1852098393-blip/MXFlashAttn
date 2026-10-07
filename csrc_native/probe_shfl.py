"""shuffle 行为探针：确认 MACA 64-lane warp 上 indexed shuffle src>=32 的行为。"""
import os, importlib.util
import torch

SRC = "/root/MXFlashAttn_v1.3.0-kernel/csrc_native"

def load_ext(name, so):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, so))
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    return ext

m1 = load_ext("mxmmain", "mxmmain.so")

inp = torch.arange(64, dtype=torch.int32, device="cuda") * 1000
stream = torch.cuda.current_stream().cuda_stream
out = m1.shfl_probe(inp, stream).cpu().numpy()

inList = inp.cpu().numpy()
ok_a = all(out[t, 0] == 33000 for t in range(64))
exp_b = [((t + 1) & 63) * 1000 for t in range(64)]
ok_b = all(out[t, 1] == exp_b[t] for t in range(64))
ok_c = all(out[t, 2] == inList[t ^ 16] for t in range(64))
ok_d = all(out[t, 3] == inList[t ^ 32] for t in range(64))

print("[A] indexed src=33        :", "OK(=33000)" if ok_a else "BAD " + str(sorted(set(out[:, 0]))[:4]))
print("[B] indexed src=(lane+1)&63:", "OK" if ok_b else "BAD " + str([out[t, 1] for t in range(30, 36)]))
print("[C] xor off=16            :", "OK" if ok_c else "BAD " + str([out[t, 2] for t in range(14, 18)]))
print("[D] xor off=32            :", "OK" if ok_d else "BAD " + str([out[t, 3] for t in range(30, 34)]))
print("SHFL-PROBE-DONE", flush=True)
